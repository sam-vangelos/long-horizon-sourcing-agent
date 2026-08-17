"""Source-neutral hypothesis / slice / lane unification layer (P1).

Adapters bridge existing ``RetrievalDesign`` / ``family_key`` substrates to the
sourcing-judgment kernel contracts without changing LinkedIn execution behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from shared.retrieval_design import (
    RetrievalDesign,
    RetrievalEdgeCaseHypothesis,
    RetrievalFamily,
    RetrievalLayerItem,
)
from shared.search_memory import normalize_family_key

if TYPE_CHECKING:
    from linkedin.search_intelligence import LinkedInExperimentState, LinkedInSearchIntent
    from shared.schemas import ExecutionPlan, SearchString

# ---------------------------------------------------------------------------
# Enum-like constants
# ---------------------------------------------------------------------------

CONSTRAINT_OPERATORS: frozenset[str] = frozenset(
    {"require", "prefer", "exclude", "avoid", "ignore"}
)
EXECUTION_SURFACES: frozenset[str] = frozenset(
    {
        "boolean_keyword",
        "boolean_title",
        "linkedin_title_filter",
        "linkedin_company_filter",
        "linkedin_location_filter",
        "soft_hint",
        "source_native",
    }
)
EXPANSION: frozenset[str] = frozenset(
    {"literal", "synonym_set", "semantic_neighborhood", "creative"}
)
TEMPORAL_SCOPES: frozenset[str] = frozenset({"current", "recent", "any"})
ACQUISITION_MODES: frozenset[str] = frozenset(
    {
        "linkedin_boolean",
        "linkedin_advanced_search",
        "linkedin_hybrid",
        "linkedin_public",
        "xray",
        "github",
        "researcher",
        "designer",
        "exec_search",
    }
)
FALLBACK_ACQUISITION_MODES: frozenset[str] = frozenset({"linkedin_public", "xray"})
SEARCH_POSTURES: frozenset[str] = frozenset(
    {"boolean_led", "filter_led", "hybrid", "structured_only"}
)
SLICE_STATUSES: frozenset[str] = frozenset(
    {"planned", "probing", "committed", "exhausted", "abandoned"}
)
AMBIGUITY_MODES: frozenset[str] = frozenset({"preserve", "resolve", "reject"})
ADAPTATION_ACTIONS: frozenset[str] = frozenset(
    {"broaden", "narrow", "rescue", "abandon", "continue", "commit", "split"}
)
PROBE_DECISIONS: frozenset[str] = frozenset(
    {"continue", "narrow", "broaden", "abandon", "mutate", "promote"}
)

DEFAULT_ACQUISITION_MODE = "linkedin_boolean"
DEFAULT_SEARCH_POSTURE = "boolean_led"
DEFAULT_SLICE_STATUS = "planned"


def _filter_dataclass_fields(cls: type, payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    return {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _dict_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldDiagnostic:
    path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message}


def normalize_lane_id(value: str | None) -> str:
    """Normalize an explicit lane identifier (no boolean anchor fallback)."""
    if not value or not str(value).strip():
        return ""
    return normalize_family_key(str(value).strip())


def validate_lane_identity_alignment(
    lane_id: str,
    family_key: str,
    *,
    path: str = "",
) -> list[FieldDiagnostic]:
    if not lane_id or not family_key:
        return []
    normalized_lane = normalize_lane_id(lane_id)
    normalized_family = normalize_lane_id(family_key)
    if normalized_lane != normalized_family:
        return [
            FieldDiagnostic(
                path=path or "lane_identity",
                code="lane_family_drift",
                message=(
                    f"lane_id {normalized_lane!r} does not match "
                    f"family_key {normalized_family!r}"
                ),
            )
        ]
    return []


def align_lane_identity(*, lane_id: str = "", family_key: str = "") -> tuple[str, str]:
    raw = (lane_id or family_key or "").strip()
    if not raw:
        return "", ""
    canonical = normalize_lane_id(raw)
    return canonical, canonical


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SearchConstraint:
    dimension: str
    values: list[str] = field(default_factory=list)
    rationale: str = ""
    operator: str = "prefer"
    execution_surface: str = "soft_hint"
    expansion: str = "literal"
    temporal_scope: str = "any"
    signal_weight: float = 0.5
    confidence: float = 0.5
    anchor_intensity: str = "medium"
    literalness: str = "literal"
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SearchConstraint:
        data = _filter_dataclass_fields(cls, payload)
        data["values"] = _string_list(data.get("values", []))
        return cls(**data)


@dataclass
class SearchHypothesis:
    hypothesis_id: str
    label: str
    target_archetype: str
    why_this_pool_may_exist: str
    capability_signals: list[str] = field(default_factory=list)
    hidden_pool_risks: list[str] = field(default_factory=list)
    source: str = "brief"
    supporting_run_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SearchHypothesis:
        data = _filter_dataclass_fields(cls, payload)
        data["capability_signals"] = _string_list(data.get("capability_signals", []))
        data["hidden_pool_risks"] = _string_list(data.get("hidden_pool_risks", []))
        data["supporting_run_refs"] = _string_list(data.get("supporting_run_refs", []))
        return cls(**data)


@dataclass
class SearchSlice:
    slice_id: str
    hypothesis_id: str
    label: str
    objective: str
    constraints: list[SearchConstraint] = field(default_factory=list)
    priority: int = 50
    status: str = DEFAULT_SLICE_STATUS
    budget: dict[str, Any] = field(default_factory=dict)
    success_metric: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["constraints"] = [item.to_dict() for item in self.constraints]
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SearchSlice:
        data = _filter_dataclass_fields(cls, payload)
        raw_constraints = payload.get("constraints", []) if payload else []
        data["constraints"] = [
            SearchConstraint.from_dict(item)
            for item in raw_constraints
            if isinstance(item, dict)
        ]
        data["budget"] = _dict_copy(data.get("budget", {}))
        data["success_metric"] = _dict_copy(data.get("success_metric", {}))
        return cls(**data)


@dataclass
class LaneExecution:
    lane_id: str
    source: str
    acquisition_mode: str = DEFAULT_ACQUISITION_MODE
    search_posture: str = DEFAULT_SEARCH_POSTURE
    boolean_strategy: dict[str, Any] = field(default_factory=dict)
    structured_filters: dict[str, Any] = field(default_factory=dict)
    recovery_policy: dict[str, Any] = field(default_factory=dict)
    ui_brittleness: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> LaneExecution:
        data = _filter_dataclass_fields(cls, payload)
        data["boolean_strategy"] = _dict_copy(data.get("boolean_strategy", {}))
        data["structured_filters"] = _dict_copy(data.get("structured_filters", {}))
        data["recovery_policy"] = _dict_copy(data.get("recovery_policy", {}))
        data["ui_brittleness"] = _dict_copy(data.get("ui_brittleness", {}))
        return cls(**data)


@dataclass
class SourcingLane:
    """Compatibility envelope: SearchHypothesis + SearchSlice + LaneExecution."""

    lane_id: str
    lane_name: str
    hypothesis: SearchHypothesis
    slice: SearchSlice
    execution: LaneExecution
    ambiguity_policy: dict[str, Any] = field(default_factory=dict)
    evidence_policy: dict[str, Any] = field(default_factory=dict)
    probe_budget: dict[str, Any] = field(default_factory=dict)
    adaptation_options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "lane_name": self.lane_name,
            "hypothesis": self.hypothesis.to_dict(),
            "slice": self.slice.to_dict(),
            "execution": self.execution.to_dict(),
            "ambiguity_policy": dict(self.ambiguity_policy),
            "evidence_policy": dict(self.evidence_policy),
            "probe_budget": dict(self.probe_budget),
            "adaptation_options": list(self.adaptation_options),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SourcingLane:
        payload = payload or {}
        return cls(
            lane_id=str(payload.get("lane_id", "")),
            lane_name=str(payload.get("lane_name", "")),
            hypothesis=SearchHypothesis.from_dict(payload.get("hypothesis")),
            slice=SearchSlice.from_dict(payload.get("slice")),
            execution=LaneExecution.from_dict(payload.get("execution")),
            ambiguity_policy=_dict_copy(payload.get("ambiguity_policy")),
            evidence_policy=_dict_copy(payload.get("evidence_policy")),
            probe_budget=_dict_copy(payload.get("probe_budget")),
            adaptation_options=_string_list(payload.get("adaptation_options", [])),
        )


@dataclass
class LaneProbe:
    probe_id: str
    lane_id: str
    work_unit_source_id: str = ""
    result_window_size: int = 20
    page_limit: int = 1
    control_snapshot: dict[str, Any] = field(default_factory=dict)
    observed_metrics: dict[str, Any] = field(default_factory=dict)
    decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> LaneProbe:
        data = _filter_dataclass_fields(cls, payload)
        data["control_snapshot"] = _dict_copy(data.get("control_snapshot", {}))
        data["observed_metrics"] = _dict_copy(data.get("observed_metrics", {}))
        return cls(**data)


VARIANT_KINDS: frozenset[str] = frozenset({
    "original", "keyword_focus", "precision", "recall",
    "structured_filter", "noise_exclusion", "rescue",
})

RESULT_WINDOW_HEALTH_STATES: frozenset[str] = frozenset({
    "too_narrow", "too_broad", "noisy", "misleading", "healthy",
})


@dataclass
class LaneVariant:
    """Source-neutral variant intent and policy for a lane.

    LinkedInSearchVariant remains the LinkedIn-specific execution tracker;
    LaneVariant captures the shared planning vocabulary that compiles down
    to source-specific state.
    """

    variant_id: str
    lane_id: str
    variant_kind: str = "original"
    hypothesis: str = ""
    status: str = "planned"
    reason: str = ""
    boolean_intent: str = ""
    structured_controls: dict[str, Any] = field(default_factory=dict)
    target_result_min: int | None = None
    target_result_max: int | None = None
    probe_budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> LaneVariant:
        data = _filter_dataclass_fields(cls, payload)
        data["structured_controls"] = _dict_copy(data.get("structured_controls", {}))
        data["probe_budget"] = _dict_copy(data.get("probe_budget", {}))
        return cls(**data)


@dataclass
class LaneMetrics:
    lane_id: str
    result_count: int = 0
    candidates_seen: int = 0
    facial_yes: int = 0
    facial_no: int = 0
    facial_borderline: int = 0
    ambiguous_high_priority: int = 0
    full_eval_count: int = 0
    save_count: int = 0
    reject_count: int = 0
    spot_check_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> LaneMetrics:
        return cls(**_filter_dataclass_fields(cls, payload))


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------


def _invalid_enum_diagnostic(
    path: str,
    field_name: str,
    value: str,
    allowed: frozenset[str],
) -> FieldDiagnostic | None:
    if value and value not in allowed:
        return FieldDiagnostic(
            path=f"{path}.{field_name}" if path else field_name,
            code=f"invalid_{field_name}",
            message=f"{field_name!r} must be one of {sorted(allowed)}; got {value!r}",
        )
    return None


def validate_search_constraint(
    constraint: SearchConstraint,
    *,
    path: str = "constraint",
) -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    if not constraint.dimension.strip():
        issues.append(
            FieldDiagnostic(path=f"{path}.dimension", code="missing_dimension", message="dimension is required")
        )
    for field_name, allowed in (
        ("operator", CONSTRAINT_OPERATORS),
        ("execution_surface", EXECUTION_SURFACES),
        ("expansion", EXPANSION),
        ("temporal_scope", TEMPORAL_SCOPES),
    ):
        diag = _invalid_enum_diagnostic(path, field_name, getattr(constraint, field_name), allowed)
        if diag:
            issues.append(diag)
    return issues


def validate_search_hypothesis(
    hypothesis: SearchHypothesis,
    *,
    path: str = "hypothesis",
) -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    for field_name in ("hypothesis_id", "label", "target_archetype", "why_this_pool_may_exist"):
        if not getattr(hypothesis, field_name, "").strip():
            issues.append(
                FieldDiagnostic(
                    path=f"{path}.{field_name}",
                    code=f"missing_{field_name}",
                    message=f"{field_name} is required",
                )
            )
    return issues


def validate_search_slice(
    search_slice: SearchSlice,
    *,
    path: str = "slice",
) -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    for field_name in ("slice_id", "hypothesis_id", "label", "objective"):
        if not getattr(search_slice, field_name, "").strip():
            issues.append(
                FieldDiagnostic(
                    path=f"{path}.{field_name}",
                    code=f"missing_{field_name}",
                    message=f"{field_name} is required",
                )
            )
    diag = _invalid_enum_diagnostic(path, "status", search_slice.status, SLICE_STATUSES)
    if diag:
        issues.append(diag)
    for index, constraint in enumerate(search_slice.constraints):
        issues.extend(
            validate_search_constraint(constraint, path=f"{path}.constraints[{index}]")
        )
    return issues


def validate_lane_execution(
    execution: LaneExecution,
    *,
    path: str = "execution",
) -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    if not execution.lane_id.strip():
        issues.append(
            FieldDiagnostic(path=f"{path}.lane_id", code="missing_lane_id", message="lane_id is required")
        )
    if not execution.source.strip():
        issues.append(
            FieldDiagnostic(path=f"{path}.source", code="missing_source", message="source is required")
        )
    for field_name, allowed in (
        ("acquisition_mode", ACQUISITION_MODES),
        ("search_posture", SEARCH_POSTURES),
    ):
        diag = _invalid_enum_diagnostic(path, field_name, getattr(execution, field_name), allowed)
        if diag:
            issues.append(diag)
    return issues


def validate_sourcing_lane(
    lane: SourcingLane,
    *,
    path: str = "lane",
) -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    if not lane.lane_id.strip():
        issues.append(
            FieldDiagnostic(path=f"{path}.lane_id", code="missing_lane_id", message="lane_id is required")
        )
    issues.extend(validate_lane_identity_alignment(lane.lane_id, lane.execution.lane_id, path=f"{path}.execution"))
    issues.extend(validate_search_hypothesis(lane.hypothesis, path=f"{path}.hypothesis"))
    issues.extend(validate_search_slice(lane.slice, path=f"{path}.slice"))
    issues.extend(validate_lane_execution(lane.execution, path=f"{path}.execution"))
    if lane.slice.hypothesis_id and lane.hypothesis.hypothesis_id:
        if normalize_lane_id(lane.slice.hypothesis_id) != normalize_lane_id(lane.hypothesis.hypothesis_id):
            issues.append(
                FieldDiagnostic(
                    path=f"{path}.slice.hypothesis_id",
                    code="hypothesis_slice_mismatch",
                    message="slice.hypothesis_id must match hypothesis.hypothesis_id",
                )
            )
    return issues


def validate_lane_probe(probe: LaneProbe, *, path: str = "probe") -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    for field_name in ("probe_id", "lane_id"):
        if not getattr(probe, field_name, "").strip():
            issues.append(
                FieldDiagnostic(
                    path=f"{path}.{field_name}",
                    code=f"missing_{field_name}",
                    message=f"{field_name} is required",
                )
            )
    diag = _invalid_enum_diagnostic(path, "decision", probe.decision, PROBE_DECISIONS)
    if diag and probe.decision:
        issues.append(diag)
    return issues


def validate_lane_metrics(metrics: LaneMetrics, *, path: str = "metrics") -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    if not metrics.lane_id.strip():
        issues.append(
            FieldDiagnostic(path=f"{path}.lane_id", code="missing_lane_id", message="lane_id is required")
        )
    return issues


def _missing_field_diagnostic(path: str, field_name: str) -> FieldDiagnostic:
    return FieldDiagnostic(
        path=path,
        code=f"missing_{field_name}",
        message=f"{field_name} is required",
    )


def validate_search_hypothesis_dict(
    payload: dict[str, Any] | None,
    *,
    path: str = "hypothesis",
) -> list[FieldDiagnostic]:
    payload = payload if isinstance(payload, dict) else {}
    issues: list[FieldDiagnostic] = []
    for field_name in ("hypothesis_id", "label", "target_archetype", "why_this_pool_may_exist"):
        if not str(payload.get(field_name, "") or "").strip():
            issues.append(_missing_field_diagnostic(f"{path}.{field_name}", field_name))
    return issues


def validate_search_slice_dict(
    payload: dict[str, Any] | None,
    *,
    path: str = "slice",
) -> list[FieldDiagnostic]:
    payload = payload if isinstance(payload, dict) else {}
    issues: list[FieldDiagnostic] = []
    for field_name in ("slice_id", "hypothesis_id", "label", "objective"):
        if not str(payload.get(field_name, "") or "").strip():
            issues.append(_missing_field_diagnostic(f"{path}.{field_name}", field_name))
    status = str(payload.get("status", "") or "").strip()
    diag = _invalid_enum_diagnostic(path, "status", status, SLICE_STATUSES)
    if diag and status:
        issues.append(diag)
    raw_constraints = payload.get("constraints", [])
    if raw_constraints is None:
        issues.append(_missing_field_diagnostic(f"{path}.constraints", "constraints"))
    elif isinstance(raw_constraints, list):
        for index, item in enumerate(raw_constraints):
            if not isinstance(item, dict):
                issues.append(
                    FieldDiagnostic(
                        path=f"{path}.constraints[{index}]",
                        code="invalid_constraint",
                        message="constraint must be a dict",
                    )
                )
                continue
            issues.extend(
                validate_search_constraint(SearchConstraint.from_dict(item), path=f"{path}.constraints[{index}]")
            )
    return issues


def validate_lane_execution_dict(
    payload: dict[str, Any] | None,
    *,
    path: str = "execution",
) -> list[FieldDiagnostic]:
    payload = payload if isinstance(payload, dict) else {}
    issues: list[FieldDiagnostic] = []
    if not str(payload.get("lane_id", "") or "").strip():
        issues.append(_missing_field_diagnostic(f"{path}.lane_id", "lane_id"))
    if not str(payload.get("source", "") or "").strip():
        issues.append(_missing_field_diagnostic(f"{path}.source", "source"))
    for field_name, allowed in (
        ("acquisition_mode", ACQUISITION_MODES),
        ("search_posture", SEARCH_POSTURES),
    ):
        value = str(payload.get(field_name, "") or "").strip()
        diag = _invalid_enum_diagnostic(path, field_name, value, allowed)
        if diag and value:
            issues.append(diag)
    return issues


def validate_sourcing_lane_dict(payload: dict[str, Any] | None) -> list[FieldDiagnostic]:
    """Validate a lane dict without constructing dataclasses (safe on malformed input)."""
    if not payload:
        return [
            FieldDiagnostic(
                path="lane",
                code="empty_payload",
                message="lane payload is empty",
            ),
            _missing_field_diagnostic("lane.lane_id", "lane_id"),
            _missing_field_diagnostic("lane.hypothesis", "hypothesis"),
            _missing_field_diagnostic("lane.slice", "slice"),
            _missing_field_diagnostic("lane.execution", "execution"),
        ]
    if not isinstance(payload, dict):
        return [
            FieldDiagnostic(
                path="lane",
                code="invalid_payload",
                message="lane payload must be a dict",
            )
        ]

    issues: list[FieldDiagnostic] = []
    if not str(payload.get("lane_id", "") or "").strip():
        issues.append(_missing_field_diagnostic("lane.lane_id", "lane_id"))

    for nested_key, validator in (
        ("hypothesis", validate_search_hypothesis_dict),
        ("slice", validate_search_slice_dict),
        ("execution", validate_lane_execution_dict),
    ):
        nested = payload.get(nested_key)
        if nested is None:
            issues.append(_missing_field_diagnostic(f"lane.{nested_key}", nested_key))
        elif isinstance(nested, dict):
            issues.extend(validator(nested, path=f"lane.{nested_key}"))
        else:
            issues.append(
                FieldDiagnostic(
                    path=f"lane.{nested_key}",
                    code=f"invalid_{nested_key}",
                    message=f"{nested_key} must be a dict",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Identity sync on generated strings / SearchString
# ---------------------------------------------------------------------------


def apply_lane_fields_to_generated_string(item: dict[str, Any]) -> dict[str, Any]:
    """Backfill lane_id from family_key or vice versa without overwriting conflicts."""
    lane_id = str(item.get("lane_id", "") or "").strip()
    family_key = str(item.get("family_key", "") or "").strip()
    drift = validate_lane_identity_alignment(lane_id, family_key, path="generated_string")
    if drift:
        return item
    aligned_lane, aligned_family = align_lane_identity(lane_id=lane_id, family_key=family_key)
    if aligned_lane:
        item["lane_id"] = aligned_lane
    if aligned_family:
        item["family_key"] = aligned_family
    return item


def apply_lane_fields_to_search_string(search_string: SearchString) -> None:
    """Backfill lane_id from family_key or vice versa on a SearchString."""
    lane_id = str(search_string.lane_id or "").strip()
    family_key = str(search_string.family_key or "").strip()
    if validate_lane_identity_alignment(lane_id, family_key, path="search_string"):
        return
    aligned_lane, aligned_family = align_lane_identity(lane_id=lane_id, family_key=family_key)
    if aligned_lane:
        search_string.lane_id = aligned_lane
    if aligned_family:
        search_string.family_key = aligned_family


def _structured_filters_have_content(filters: Any) -> bool:
    """Whether a structured_filters dict carries any LIVE applied facet.

    Mirrors LinkedInStructuredFilters.is_empty semantics without importing the
    LinkedIn layer into shared/: a live facet is a non-empty titles/companies/
    skills/assessments list or a non-empty sidebar/advanced bucket. ``target_employers``
    is deliberately excluded — it is family metadata the compiler folds into
    ``companies``, not a direct per-string facet.
    """
    if not isinstance(filters, dict):
        return False
    for key in ("titles", "companies", "skills", "assessments"):
        if filters.get(key):
            return True
    for bucket_key in ("sidebar_filters", "advanced_filters"):
        bucket = filters.get(bucket_key)
        if isinstance(bucket, dict) and any(bucket.values()):
            return True
    return False


def lane_fields_from_work_unit_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return SearchString lane kwargs from a generated string or coverage-gap dict."""
    working = apply_lane_fields_to_generated_string(dict(item))
    recipe = working.get("retrieval_recipe") if isinstance(working.get("retrieval_recipe"), dict) else {}
    lane_snapshot = dict(working.get("lane_snapshot") or {})
    if recipe and "retrieval_recipe" not in lane_snapshot:
        lane_snapshot = {**lane_snapshot, "retrieval_recipe": dict(recipe)}
    # P2 hop 2: lift the lane compiler's structured_filters (parked in the compiler
    # snapshot by apply_linkedin_lane_compiler_to_plan) onto a first-class
    # SearchString field so execution can read it without spelunking lane_snapshot.
    compiler_snapshot = lane_snapshot.get("compiler")
    compiler_snapshot = compiler_snapshot if isinstance(compiler_snapshot, dict) else {}
    query_payload = compiler_snapshot.get("query_payload")
    query_payload = query_payload if isinstance(query_payload, dict) else {}
    structured_filters = query_payload.get("structured_filters")
    structured_filters = dict(structured_filters) if isinstance(structured_filters, dict) else {}
    # Production unlock (Slice 1): when no lane-compiler snapshot supplied filters, honor
    # the work unit's OWN top-level structured_filters. This lets a generated string /
    # coverage gap carry a per-string title/company/location filter directly to execution
    # without depending on a fragile family_key -> hybrid-lane projection match. Default
    # path is byte-identical: today's strings carry no top-level structured_filters, so
    # this is a no-op until the producer is taught to emit them.
    if not _structured_filters_have_content(structured_filters):
        own_filters = working.get("structured_filters")
        if _structured_filters_have_content(own_filters):
            structured_filters = dict(own_filters)
    lane_name = str(working.get("lane_name") or recipe.get("family_label") or "").strip()
    lane_intent = str(
        working.get("lane_intent")
        or working.get("rationale")
        or working.get("gap")
        or ""
    ).strip()
    acquisition_mode = str(
        working.get("acquisition_mode")
        or compiler_snapshot.get("acquisition_mode")
        or DEFAULT_ACQUISITION_MODE
    ).strip()
    # A non-empty structured filter set forces hybrid acquisition (the apply layer
    # rejects a boolean variant that carries structured filters, and the opening gate
    # only fires the structured apply when acquisition_mode == linkedin_hybrid). Only
    # promote from the default boolean mode; an explicitly-declared mode is preserved.
    if acquisition_mode == DEFAULT_ACQUISITION_MODE and _structured_filters_have_content(
        structured_filters
    ):
        acquisition_mode = "linkedin_hybrid"
    search_posture = str(
        working.get("search_posture")
        or compiler_snapshot.get("search_posture")
        or DEFAULT_SEARCH_POSTURE
    ).strip()
    return {
        "lane_id": str(working.get("lane_id", "") or ""),
        "lane_name": lane_name,
        "lane_intent": lane_intent,
        "acquisition_mode": acquisition_mode,
        "search_posture": search_posture,
        "lane_snapshot": lane_snapshot,
        "structured_filters": structured_filters,
        "boolean_normalization": (
            dict(working.get("boolean_normalization"))
            if isinstance(working.get("boolean_normalization"), dict)
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# Retrieval design adapters
# ---------------------------------------------------------------------------


def _layer_terms(items: list[RetrievalLayerItem], *, limit: int | None = None) -> list[str]:
    terms: list[str] = []
    for item in items:
        if not item.enabled:
            continue
        for term in item.terms:
            text = str(term or "").strip()
            if text and text not in terms:
                terms.append(text)
            if limit and len(terms) >= limit:
                return terms
    return terms


def derive_search_posture(constraints_or_surfaces: list[Any]) -> str:
    """Derive a SEARCH_POSTURES label from a lane's constraints (or their surfaces).

    TELEMETRY ONLY (Slice A part 5). Accepts either SearchConstraint objects or raw
    execution_surface strings. Classifies each *positive-execution* surface as Boolean
    (``boolean_*``) or structured filter (``linkedin_*_filter``); ``soft_hint`` and any
    other planner-metadata surface carry no posture signal and are ignored.

    - only Boolean surfaces (or no signalling surface) -> "boolean_led" (DEFAULT)
    - only structured filter surfaces                  -> "structured_only"
    - a mix of both                                    -> "hybrid"

    Never returns "filter_led" — that posture is reserved for the planner's own
    declaration, not derivable from surface composition alone. This value is attached
    as telemetry and MUST NOT influence acquisition_mode (the lane compiler derives mode
    independently from structured.is_empty()).
    """
    has_boolean = False
    has_filter = False
    for item in constraints_or_surfaces or []:
        surface = item if isinstance(item, str) else getattr(item, "execution_surface", "")
        surface = str(surface or "")
        if surface.startswith("boolean_"):
            has_boolean = True
        elif surface.startswith("linkedin_") and surface.endswith("_filter"):
            has_filter = True
    if has_boolean and has_filter:
        return "hybrid"
    if has_filter:
        return "structured_only"
    return DEFAULT_SEARCH_POSTURE


def search_constraints_from_layer_items(
    items: list[RetrievalLayerItem],
    *,
    dimension: str,
    default_operator: str = "prefer",
) -> list[SearchConstraint]:
    constraints: list[SearchConstraint] = []
    for item in items:
        if not item.enabled:
            continue
        terms = _layer_terms([item])
        if not terms:
            continue
        operator = "exclude" if dimension == "anti_noise" else default_operator
        # build #2: a layer item may opt into a structured title/company filter surface.
        # anti_noise NEVER emits a structured filter — negative structured filtering is
        # not wired (the positive-only fold in lane_compiler would silently drop it), so
        # it stays soft_hint. Default "" -> boolean_keyword: byte-identical to before.
        item_surface = getattr(item, "structured_surface", "") or ""
        if dimension == "anti_noise":
            surface = "soft_hint"
        elif item_surface:
            surface = item_surface
        else:
            surface = "boolean_keyword"
        constraints.append(
            SearchConstraint(
                dimension=dimension,
                values=terms,
                rationale=item.rationale or item.label,
                operator=operator,
                execution_surface=surface,
            )
        )
    return constraints


def search_hypothesis_from_retrieval_family(family: RetrievalFamily) -> SearchHypothesis:
    capability_signals = _layer_terms(family.capability_proxies)
    hidden_pool_risks = _layer_terms(family.anti_noise)
    if family.notes:
        hidden_pool_risks = list(dict.fromkeys(hidden_pool_risks + [family.notes]))
    return SearchHypothesis(
        hypothesis_id=normalize_lane_id(family.family_id),
        label=family.label,
        target_archetype=family.label,
        why_this_pool_may_exist=family.objective or family.label,
        capability_signals=capability_signals,
        hidden_pool_risks=hidden_pool_risks,
        source="retrieval_family",
    )


def search_hypothesis_from_edge_case(
    hypothesis: RetrievalEdgeCaseHypothesis,
) -> SearchHypothesis:
    capability_signals: list[str] = []
    for layer in (
        hypothesis.entry_signal_variants,
        hypothesis.capability_proxy_variants,
        hypothesis.reality_filter_variants,
        hypothesis.context_constraint_variants,
    ):
        capability_signals.extend(_layer_terms(layer))
    capability_signals = list(dict.fromkeys(capability_signals))
    hidden_pool_risks = list(hypothesis.noise_risks)
    if hypothesis.validation_rule:
        hidden_pool_risks.append(hypothesis.validation_rule)
    return SearchHypothesis(
        hypothesis_id=normalize_lane_id(hypothesis.hypothesis_id),
        label=hypothesis.label,
        target_archetype=hypothesis.hidden_cohort,
        why_this_pool_may_exist=hypothesis.why_missed,
        capability_signals=capability_signals,
        hidden_pool_risks=hidden_pool_risks,
        source=hypothesis.source or "edge_case",
        supporting_run_refs=list(hypothesis.supporting_run_refs),
    )


def search_slice_from_retrieval_family(family: RetrievalFamily) -> SearchSlice:
    lane_id = normalize_lane_id(family.family_id)
    constraints: list[SearchConstraint] = []
    for dimension, items, operator in (
        ("entry_signal", family.entry_signals, "prefer"),
        ("capability", family.capability_proxies, "prefer"),
        ("reality", family.reality_filters, "require"),
        ("context", family.context_constraints, "prefer"),
        ("anti_noise", family.anti_noise, "exclude"),
    ):
        constraints.extend(
            search_constraints_from_layer_items(
                items,
                dimension=dimension,
                default_operator=operator,
            )
        )
    return SearchSlice(
        slice_id=f"{lane_id}_slice",
        hypothesis_id=lane_id,
        label=family.label,
        objective=family.objective or family.label,
        constraints=constraints,
        priority=family.priority,
        status=DEFAULT_SLICE_STATUS if family.enabled else "abandoned",
    )


def lane_execution_from_retrieval_family(
    family: RetrievalFamily,
    *,
    acquisition_mode: str = DEFAULT_ACQUISITION_MODE,
    source: str = "linkedin",
) -> LaneExecution:
    lane_id = normalize_lane_id(family.family_id)
    return LaneExecution(
        lane_id=lane_id,
        source=source,
        acquisition_mode=acquisition_mode,
        search_posture=DEFAULT_SEARCH_POSTURE,
        boolean_strategy={"family_label": family.label},
        structured_filters={
            "target_employers": list(family.target_employers),
            "target_markets": list(family.target_markets),
        },
    )


def sourcing_lane_from_retrieval_family(
    family: RetrievalFamily,
    *,
    acquisition_mode: str = DEFAULT_ACQUISITION_MODE,
    source: str = "linkedin",
) -> SourcingLane:
    hypothesis = search_hypothesis_from_retrieval_family(family)
    search_slice = search_slice_from_retrieval_family(family)
    execution = lane_execution_from_retrieval_family(
        family,
        acquisition_mode=acquisition_mode,
        source=source,
    )
    execution.search_posture = derive_search_posture(search_slice.constraints)
    lane_id = normalize_lane_id(family.family_id)
    return SourcingLane(
        lane_id=lane_id,
        lane_name=family.label,
        hypothesis=hypothesis,
        slice=search_slice,
        execution=execution,
        ambiguity_policy={"mode": "preserve"},
        probe_budget={"variants_to_emit": family.variants_to_emit},
        adaptation_options=list(ADAPTATION_ACTIONS)[:3],
    )


def sourcing_lanes_from_retrieval_design(design: RetrievalDesign) -> list[SourcingLane]:
    lanes: list[SourcingLane] = []
    for family in design.families:
        if not family.enabled:
            continue
        lanes.append(sourcing_lane_from_retrieval_family(family))
    return lanes


def populate_execution_plan_lane_payloads(plan: ExecutionPlan) -> None:
    """Fill lane payload lists from retrieval_families without touching generated_strings.

    The lane payloads (search_hypotheses / search_slices / sourcing_lanes,
    including the compiled query_payload under them) are planner-metadata
    reconstructions from retrieval_families and are never submitted to the
    browser. The executed boolean derives from search_string.boolean and is
    recorded by the string_executed event; string_started keeps queue-time
    semantics.
    """
    if not plan.retrieval_families:
        return
    design = RetrievalDesign.from_dict({"families": plan.retrieval_families})
    lanes = sourcing_lanes_from_retrieval_design(design)
    plan.search_hypotheses = [lane.hypothesis.to_dict() for lane in lanes]
    plan.search_slices = [lane.slice.to_dict() for lane in lanes]
    plan.sourcing_lanes = [lane.to_dict() for lane in lanes]


# ---------------------------------------------------------------------------
# LinkedIn experiment bridge (read-only)
# ---------------------------------------------------------------------------


def lane_id_from_search_intent(intent: LinkedInSearchIntent) -> str:
    return normalize_lane_id(intent.family_key)


def lane_id_from_experiment_state(state: LinkedInExperimentState) -> str:
    return lane_id_from_search_intent(state.intent)


# ---------------------------------------------------------------------------
# Planner feedback diffs (D3)
# ---------------------------------------------------------------------------

PLANNER_DIFF_ACTIONS: frozenset[str] = frozenset({"add", "update", "retire"})
PLANNER_DIFF_TARGET_TYPES: frozenset[str] = frozenset({
    "hypothesis", "slice", "constraint", "variant", "validation_question",
})


@dataclass(frozen=True)
class PlannerDiff:
    """Executable planner diff produced by market-intelligence feedback.

    Each diff targets a specific planner type (hypothesis/slice/constraint/variant)
    with an action (add/update/retire). Non-compilable recommendations are dropped
    from planner output and preserved in the side report.
    """

    diff_id: str
    action: str
    target_type: str
    target_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    internal_evidence: list[str] = field(default_factory=list)
    external_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> PlannerDiff:
        data = _filter_dataclass_fields(cls, payload)
        data["payload"] = _dict_copy(data.get("payload", {}))
        data["internal_evidence"] = _string_list(data.get("internal_evidence", []))
        data["external_evidence"] = _string_list(data.get("external_evidence", []))
        return cls(**data)

    def is_valid(self) -> bool:
        return bool(
            self.diff_id
            and self.action in PLANNER_DIFF_ACTIONS
            and self.target_type in PLANNER_DIFF_TARGET_TYPES
        )


def validate_planner_diff(diff: PlannerDiff, *, path: str = "diff") -> list[FieldDiagnostic]:
    issues: list[FieldDiagnostic] = []
    if not diff.diff_id:
        issues.append(FieldDiagnostic(path=f"{path}.diff_id", code="missing_diff_id", message="diff_id is required"))
    diag = _invalid_enum_diagnostic(path, "action", diff.action, PLANNER_DIFF_ACTIONS)
    if diag:
        issues.append(diag)
    diag = _invalid_enum_diagnostic(path, "target_type", diff.target_type, PLANNER_DIFF_TARGET_TYPES)
    if diag:
        issues.append(diag)
    return issues
