"""Runtime-backed LinkedIn search-intelligence state and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from linkedin.page_allocator import PageObservation
from shared import config
from shared.schemas import SearchString

if TYPE_CHECKING:
    from linkedin.search_intelligence import LinkedInStructuredFilters


def result_window_for_count(result_count: int) -> tuple[int, int] | None:
    """Return the target result window for a noisy result set."""
    if result_count > 5000:
        return (200, 1200)
    if result_count >= 1500:
        return (150, 800)
    if result_count >= 500:
        return (75, 400)
    return None


def _is_filter_led_surface(
    surface: str,
    structured_filters: "LinkedInStructuredFilters | None",
) -> bool:
    """A variant is filter-led when its surface is hybrid/structured_only OR it
    carries any structured filter. Mirrors the discriminator slices C–E key on
    (variant.surface in {hybrid, structured_only}, non-empty structured_filters)."""
    if surface in {"hybrid", "structured_only"}:
        return True
    return structured_filters is not None and not structured_filters.is_empty()


def scale_window_for_surface(
    target_result_min: int | None,
    target_result_max: int | None,
    *,
    surface: str,
    structured_filters: "LinkedInStructuredFilters | None",
) -> tuple[int | None, int | None]:
    """Phase 2 hop 4 (slice F): scale a healthy result-window DOWN for a filter-led
    variant AT CONSTRUCTION.

    A filter-led / structured search is legitimately narrower than a keyword search,
    so the keyword-tuned lifecycle gate (classify_result_window) would mis-read a good
    structured probe as too_narrow and abandon it. Scaling the window here — at the
    variant build sites — bakes the posture into target_result_min/max so the decision
    functions stay PURE: classify_result_window / decide_variant_lifecycle read an
    already-scaled window and never see surface or the factor.

    A BOOLEAN variant (surface not hybrid/structured_only AND empty filters) is
    returned UNCHANGED — the default path is byte-identical to pre-F. A None bound
    (e.g. the root variant before a window is proposed) passes through untouched.

    Floor + ordering guards: a scaled min stays >= 1 and min <= max, so a tiny window
    cannot collapse to 0 or invert.
    """
    if target_result_min is None or target_result_max is None:
        return target_result_min, target_result_max
    if not _is_filter_led_surface(surface, structured_filters):
        return target_result_min, target_result_max
    factor = config.SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR
    scaled_min = max(1, round(target_result_min * factor))
    scaled_max = max(scaled_min, round(target_result_max * factor))
    return scaled_min, scaled_max


def unscale_window_from_surface(
    target_result_min: int | None,
    target_result_max: int | None,
) -> tuple[int | None, int | None]:
    """Phase 2 hop 4 (slice F): project a filter-led (already-scaled) window back UP to
    the keyword baseline — the inverse of scale_window_for_surface.

    Used only on a posture-FLIP rescue (filter-led parent -> boolean child): a
    broaden/recall rescue that drops the parent's last filter leaves a runnable keyword
    child, but spawn_rescue_variant_from_hint inherits the parent's narrow structured
    window. Without re-projection the keyword child is judged by the keyword-tuned gate
    (classify_result_window) against a structured-scaled window — the directional inverse
    of the structured-narrow dead-end slice F closed. Dividing by the same factor restores
    keyword-class bounds so a healthy keyword count is not mis-read as too_broad.

    A None bound passes through. Floor + ordering guards mirror the down-scale: min >= 1,
    min <= max. Round-trip is intentionally not byte-exact (rounding is not invertible),
    so the SAME-posture path inherits the parent window raw and never calls this — only a
    genuine flip-up does, where keyword-class magnitude (not byte-equality) is what matters.
    """
    if target_result_min is None or target_result_max is None:
        return target_result_min, target_result_max
    factor = config.SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR
    if factor <= 0:
        return target_result_min, target_result_max
    unscaled_min = max(1, round(target_result_min / factor))
    unscaled_max = max(unscaled_min, round(target_result_max / factor))
    return unscaled_min, unscaled_max


def _surface_for_rescue_child(
    boolean: str,
    structured_filters: "LinkedInStructuredFilters | None",
) -> str:
    """Resolve a rescue child's surface from its RESOLVED (boolean, filters), mirroring
    the slice C–E discriminator (LinkedInSearchVariant.surface doc, :324-330).

    The rescue spawner has no LLM proposal to read a surface off of, so it derives one:
    filters present + boolean -> 'hybrid'; filters present + no boolean -> 'structured_only';
    boolean only -> 'boolean'; neither -> '' (the keyword default — an un-runnable child the
    spawner rejects upstream). This lets the spawner set a truthful surface AND route the
    inherited window through the same posture-aware derivation the four build sites use,
    instead of defaulting surface='' onto a child that may have flipped posture.
    """
    has_filters = structured_filters is not None and not structured_filters.is_empty()
    has_boolean = bool(boolean.strip())
    if has_filters:
        return "hybrid" if has_boolean else "structured_only"
    if has_boolean:
        return "boolean"
    return ""


def reproject_rescue_window(
    target_result_min: int | None,
    target_result_max: int | None,
    *,
    parent_surface: str,
    parent_filters: "LinkedInStructuredFilters | None",
    child_surface: str,
    child_filters: "LinkedInStructuredFilters | None",
) -> tuple[int | None, int | None]:
    """Re-project an inherited window across a rescue's parent->child posture transition.

    spawn_rescue_variant_from_hint is the fifth window-write site (the four build sites in
    orchestrator/_plan_* and bootstrap all scale_window_for_surface against the CHILD's
    resolved surface). It copies the parent window raw, which is correct ONLY when posture
    is unchanged. The two flips it silently mis-handles:

      * filter-led parent -> boolean child (broaden dropped the last filter): the inherited
        window is structured-scaled (narrow); un-scale it back to keyword bounds so the
        keyword child is not mis-read as too_broad.
      * boolean parent -> filter-led child: scale the inherited keyword window DOWN, exactly
        as the build sites do for a structured variant.

    Same posture (both filter-led, or both boolean) inherits the window RAW — no transform,
    no rounding — preserving the byte-stable behavior the same-posture rescue depends on.
    A None window passes through untouched.
    """
    if target_result_min is None or target_result_max is None:
        return target_result_min, target_result_max
    parent_filter_led = _is_filter_led_surface(parent_surface, parent_filters)
    child_filter_led = _is_filter_led_surface(child_surface, child_filters)
    if parent_filter_led == child_filter_led:
        return target_result_min, target_result_max
    if parent_filter_led and not child_filter_led:
        return unscale_window_from_surface(target_result_min, target_result_max)
    return scale_window_for_surface(
        target_result_min,
        target_result_max,
        surface=child_surface,
        structured_filters=child_filters,
    )


@dataclass
class LinkedInStructuredFilters:
    titles: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    assessments: list[str] = field(default_factory=list)
    sidebar_filters: dict[str, Any] = field(default_factory=dict)
    advanced_filters: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(
            (
                self.titles,
                self.companies,
                self.skills,
                self.assessments,
                self.sidebar_filters,
                self.advanced_filters,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "titles": list(self.titles),
            "companies": list(self.companies),
            "skills": list(self.skills),
            "assessments": list(self.assessments),
            "sidebar_filters": dict(self.sidebar_filters),
            "advanced_filters": dict(self.advanced_filters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LinkedInStructuredFilters":
        payload = payload or {}
        return cls(
            titles=list(payload.get("titles", [])),
            companies=list(payload.get("companies", [])),
            skills=list(payload.get("skills", [])),
            assessments=list(payload.get("assessments", [])),
            sidebar_filters=dict(payload.get("sidebar_filters", {})),
            advanced_filters=dict(payload.get("advanced_filters", {})),
        )


@dataclass
class LinkedInSearchIntent:
    root_boolean: str
    family_key: str = ""
    novelty_bucket: str = ""
    domain_lane: str = ""
    retrieval_recipe: dict[str, Any] = field(default_factory=dict)
    applied_hypothesis_ids: list[str] = field(default_factory=list)
    structured_filters: LinkedInStructuredFilters = field(default_factory=LinkedInStructuredFilters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_boolean": self.root_boolean,
            "family_key": self.family_key,
            "novelty_bucket": self.novelty_bucket,
            "domain_lane": self.domain_lane,
            "retrieval_recipe": dict(self.retrieval_recipe),
            "applied_hypothesis_ids": list(self.applied_hypothesis_ids),
            "structured_filters": self.structured_filters.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LinkedInSearchIntent":
        payload = payload or {}
        return cls(
            root_boolean=payload.get("root_boolean", ""),
            family_key=payload.get("family_key", ""),
            novelty_bucket=payload.get("novelty_bucket", ""),
            domain_lane=payload.get("domain_lane", ""),
            retrieval_recipe=dict(payload.get("retrieval_recipe", {})),
            applied_hypothesis_ids=list(payload.get("applied_hypothesis_ids", [])),
            structured_filters=LinkedInStructuredFilters.from_dict(payload.get("structured_filters")),
        )


@dataclass
class LinkedInPageInsights:
    page: int
    result_count: int
    result_window: str
    title_clusters: list[dict[str, Any]] = field(default_factory=list)
    company_clusters: list[dict[str, Any]] = field(default_factory=list)
    signal_anchors: list[str] = field(default_factory=list)
    noise_anchors: list[str] = field(default_factory=list)
    dominant_non_fit_patterns: list[str] = field(default_factory=list)
    glance_action: str = ""
    glance_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "result_count": self.result_count,
            "result_window": self.result_window,
            "title_clusters": list(self.title_clusters),
            "company_clusters": list(self.company_clusters),
            "signal_anchors": list(self.signal_anchors),
            "noise_anchors": list(self.noise_anchors),
            "dominant_non_fit_patterns": list(self.dominant_non_fit_patterns),
            "glance_action": self.glance_action,
            "glance_summary": self.glance_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LinkedInPageInsights | None":
        if not payload:
            return None
        return cls(
            page=int(payload.get("page", 0)),
            result_count=int(payload.get("result_count", 0)),
            result_window=str(payload.get("result_window", "")),
            title_clusters=list(payload.get("title_clusters", [])),
            company_clusters=list(payload.get("company_clusters", [])),
            signal_anchors=list(payload.get("signal_anchors", [])),
            noise_anchors=list(payload.get("noise_anchors", [])),
            dominant_non_fit_patterns=list(payload.get("dominant_non_fit_patterns", [])),
            glance_action=str(payload.get("glance_action", "")),
            glance_summary=str(payload.get("glance_summary", "")),
        )


def _full_outcome_metrics(page_stats: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return reviewed/outreach/review/reject counts with legacy fallbacks.

    New callers provide the four ``full_*`` counters. Older checkpoints only
    know physical saves and full-profile rejects, which remain conservative
    settled-outcome proxies. Raw facial positives are intentionally excluded.
    """

    outreach = int(
        page_stats.get(
            "full_outreach",
            page_stats.get("saves", 0),
        )
        or 0
    )
    review = int(page_stats.get("full_review", 0) or 0)
    reject = int(
        page_stats.get(
            "full_reject",
            page_stats.get("rejects", 0),
        )
        or 0
    )
    reviewed = int(
        page_stats.get(
            "full_reviewed",
            outreach + review + reject,
        )
        or 0
    )
    return (
        max(reviewed, outreach + review + reject),
        max(0, outreach),
        max(0, review),
        max(0, reject),
    )


@dataclass
class LinkedInVariantSnapshot:
    page_start: int
    page_end: int
    result_count: int
    result_window: str
    title_clusters: list[dict[str, Any]] = field(default_factory=list)
    company_clusters: list[dict[str, Any]] = field(default_factory=list)
    signal_anchors: list[str] = field(default_factory=list)
    noise_anchors: list[str] = field(default_factory=list)
    dominant_non_fit_patterns: list[str] = field(default_factory=list)
    signal_weight: float = 0.0
    noise_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_start": self.page_start,
            "page_end": self.page_end,
            "result_count": self.result_count,
            "result_window": self.result_window,
            "title_clusters": list(self.title_clusters),
            "company_clusters": list(self.company_clusters),
            "signal_anchors": list(self.signal_anchors),
            "noise_anchors": list(self.noise_anchors),
            "dominant_non_fit_patterns": list(self.dominant_non_fit_patterns),
            "signal_weight": self.signal_weight,
            "noise_weight": self.noise_weight,
        }

    @classmethod
    def from_page(
        cls,
        *,
        page_num: int,
        result_count: int,
        page_insights: LinkedInPageInsights,
        page_stats: dict[str, int],
    ) -> "LinkedInVariantSnapshot":
        _, outreach, review, reject = _full_outcome_metrics(page_stats)
        signal_weight = float(outreach * 3 + review)
        noise_weight = float(int(page_stats.get("facial_no", 0) or 0) + reject)
        return cls(
            page_start=page_num,
            page_end=page_num,
            result_count=result_count,
            result_window=page_insights.result_window,
            title_clusters=list(page_insights.title_clusters),
            company_clusters=list(page_insights.company_clusters),
            signal_anchors=list(page_insights.signal_anchors),
            noise_anchors=list(page_insights.noise_anchors),
            dominant_non_fit_patterns=list(page_insights.dominant_non_fit_patterns),
            signal_weight=signal_weight,
            noise_weight=noise_weight,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LinkedInVariantSnapshot | None":
        if not payload:
            return None
        return cls(
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            result_count=int(payload.get("result_count", 0)),
            result_window=str(payload.get("result_window", "")),
            title_clusters=list(payload.get("title_clusters", [])),
            company_clusters=list(payload.get("company_clusters", [])),
            signal_anchors=list(payload.get("signal_anchors", [])),
            noise_anchors=list(payload.get("noise_anchors", [])),
            dominant_non_fit_patterns=list(payload.get("dominant_non_fit_patterns", [])),
            signal_weight=float(payload.get("signal_weight", 0.0)),
            noise_weight=float(payload.get("noise_weight", 0.0)),
        )


@dataclass
class LinkedInDriftAssessment:
    decision: str
    rationale: str
    eligible: bool
    overfit_risk: str = ""
    keyword_hypothesis: str = ""
    future_filter_hypothesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "rationale": self.rationale,
            "eligible": self.eligible,
            "overfit_risk": self.overfit_risk,
            "keyword_hypothesis": self.keyword_hypothesis,
            "future_filter_hypothesis": self.future_filter_hypothesis,
        }


VARIANT_LIFECYCLE_STATUSES = frozenset({
    "planned", "probing", "active", "explored", "committed", "exhausted", "abandoned",
})


def _nonnegative_shadow_int(value: Any) -> int:
    """Parse an additive allocator counter without risking primary resume."""

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _shadow_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _shadow_observation(value: Any) -> PageObservation | None:
    if not isinstance(value, dict):
        return None
    try:
        return PageObservation.from_dict(value)
    except (AttributeError, TypeError, ValueError):
        return None


def _variant_shadow_observations(
    value: Any,
    *,
    root_string_id: int,
    variant_id: str,
) -> list[PageObservation]:
    if not isinstance(value, list):
        return []
    observations = [
        observation
        for item in value
        if (observation := _shadow_observation(item)) is not None
        and observation.root_string_id == root_string_id
        and observation.variant_id == variant_id
        and observation.teaches_policy
    ]
    return observations[-2:]


@dataclass
class LinkedInSearchVariant:
    variant_id: str
    parent_variant_id: str | None
    root_string_id: int
    boolean: str
    variant_kind: str = "original"
    hypothesis: str = ""
    target_result_min: int | None = None
    target_result_max: int | None = None
    status: str = "planned"
    experiment_round: int = 0
    structured_filters: LinkedInStructuredFilters = field(default_factory=LinkedInStructuredFilters)
    result_count: int = 0
    pages_reviewed: int = 0
    # First page not yet durably completed for this specific search variant.
    # Variant-local ownership matters because a failed drift can return to a
    # committed parent whose next page differs from the drift probe's cursor.
    allocator_page_cursor: int = 0
    # Allocator learning is local to a concrete rewrite. Completed observations
    # include valid, invalid, and off-policy pages; only valid on-policy pages
    # enter the bounded teaching window.
    allocator_valid_page_count: int = 0
    allocator_completed_observation_count: int = 0
    allocator_observations: list[PageObservation] = field(default_factory=list)
    candidates: int = 0
    duplicates: int = 0
    saves: int = 0
    rejects: int = 0
    # Settled full-profile funnel. Facial outcomes remain diagnostic only and
    # do not earn search commitment.
    full_reviewed: int = 0
    full_outreach: int = 0
    full_review: int = 0
    full_reject: int = 0
    facial_yes: int = 0
    facial_borderline: int = 0
    facial_no: int = 0
    last_page_insights: LinkedInPageInsights | None = None
    lane_id: str = ""
    lifecycle_reason: str = ""
    result_window_health: str = ""
    # Phase 2 hop 4 (slice C): how this variant executes against the page —
    # "boolean" (keyword-only entry), "hybrid" (keyword + structured filters), or
    # "structured_only" (filters carry it, no keyword). Default "" reads as the
    # legacy boolean behavior. A "boolean" surface on a variant_kind=="structured_filter"
    # is the deliberate demote-to-boolean marker the seeding gate keys on (so the
    # lane's filters are NOT re-seeded onto a variant that chose to drop them).
    surface: str = ""
    probe_page_budget: int = 1
    probe_pages_used: int = 0

    def __post_init__(self) -> None:
        self.allocator_observations = [
            observation
            for observation in self.allocator_observations
            if observation.root_string_id == self.root_string_id
            and observation.variant_id == self.variant_id
            and observation.teaches_policy
        ][-2:]
        self.allocator_valid_page_count = max(
            _nonnegative_shadow_int(self.allocator_valid_page_count),
            len(self.allocator_observations),
        )
        self.allocator_completed_observation_count = _nonnegative_shadow_int(
            self.allocator_completed_observation_count
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "variant_id": self.variant_id,
            "parent_variant_id": self.parent_variant_id,
            "root_string_id": self.root_string_id,
            "boolean": self.boolean,
            "variant_kind": self.variant_kind,
            "hypothesis": self.hypothesis,
            "target_result_min": self.target_result_min,
            "target_result_max": self.target_result_max,
            "status": self.status,
            "experiment_round": self.experiment_round,
            "structured_filters": self.structured_filters.to_dict(),
            "result_count": self.result_count,
            "pages_reviewed": self.pages_reviewed,
            "allocator_page_cursor": self.allocator_page_cursor,
            "allocator_valid_page_count": self.allocator_valid_page_count,
            "allocator_completed_observation_count": (
                self.allocator_completed_observation_count
            ),
            "allocator_observations": [
                observation.to_dict()
                for observation in self.allocator_observations[-2:]
            ],
            "candidates": self.candidates,
            "duplicates": self.duplicates,
            "saves": self.saves,
            "rejects": self.rejects,
            "full_reviewed": self.full_reviewed,
            "full_outreach": self.full_outreach,
            "full_review": self.full_review,
            "full_reject": self.full_reject,
            "facial_yes": self.facial_yes,
            "facial_borderline": self.facial_borderline,
            "facial_no": self.facial_no,
            "last_page_insights": self.last_page_insights.to_dict() if self.last_page_insights else None,
            "lane_id": self.lane_id,
            "lifecycle_reason": self.lifecycle_reason,
            "result_window_health": self.result_window_health,
            "surface": self.surface,
            "probe_page_budget": self.probe_page_budget,
            "probe_pages_used": self.probe_pages_used,
        }
        for key in ("lane_id", "lifecycle_reason", "result_window_health", "surface"):
            if d.get(key) == "":
                d.pop(key, None)
        if d.get("probe_page_budget") == 1 and d.get("probe_pages_used") == 0:
            d.pop("probe_page_budget", None)
            d.pop("probe_pages_used", None)
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinkedInSearchVariant":
        variant_id = str(payload.get("variant_id", "root"))
        root_string_id = int(payload.get("root_string_id", 0))
        full_outreach = int(payload.get("full_outreach", payload.get("saves", 0)) or 0)
        full_review = int(payload.get("full_review", 0) or 0)
        full_reject = int(payload.get("full_reject", payload.get("rejects", 0)) or 0)
        full_reviewed = int(
            payload.get(
                "full_reviewed",
                full_outreach + full_review + full_reject,
            )
            or 0
        )
        return cls(
            variant_id=variant_id,
            parent_variant_id=payload.get("parent_variant_id"),
            root_string_id=root_string_id,
            boolean=str(payload.get("boolean", "")),
            variant_kind=str(payload.get("variant_kind", "original")),
            hypothesis=str(payload.get("hypothesis", "")),
            target_result_min=payload.get("target_result_min"),
            target_result_max=payload.get("target_result_max"),
            status=str(payload.get("status", "planned")),
            experiment_round=int(payload.get("experiment_round", 0)),
            structured_filters=LinkedInStructuredFilters.from_dict(payload.get("structured_filters")),
            result_count=int(payload.get("result_count", 0)),
            pages_reviewed=int(payload.get("pages_reviewed", 0)),
            allocator_page_cursor=int(payload.get("allocator_page_cursor", 0) or 0),
            allocator_valid_page_count=_nonnegative_shadow_int(
                payload.get("allocator_valid_page_count", 0)
            ),
            allocator_completed_observation_count=_nonnegative_shadow_int(
                payload.get("allocator_completed_observation_count", 0)
            ),
            allocator_observations=_variant_shadow_observations(
                payload.get("allocator_observations", []),
                root_string_id=root_string_id,
                variant_id=variant_id,
            ),
            candidates=int(payload.get("candidates", 0)),
            duplicates=int(payload.get("duplicates", 0)),
            saves=int(payload.get("saves", 0)),
            rejects=int(payload.get("rejects", 0)),
            full_reviewed=max(full_reviewed, full_outreach + full_review + full_reject),
            full_outreach=full_outreach,
            full_review=full_review,
            full_reject=full_reject,
            facial_yes=int(payload.get("facial_yes", 0)),
            facial_borderline=int(payload.get("facial_borderline", 0)),
            facial_no=int(payload.get("facial_no", 0)),
            last_page_insights=LinkedInPageInsights.from_dict(payload.get("last_page_insights")),
            lane_id=str(payload.get("lane_id", "")),
            lifecycle_reason=str(payload.get("lifecycle_reason", "")),
            result_window_health=str(payload.get("result_window_health", "")),
            surface=str(payload.get("surface", "")),
            probe_page_budget=int(payload.get("probe_page_budget", 1)),
            probe_pages_used=int(payload.get("probe_pages_used", 0)),
        )

    def within_target_window(self) -> bool:
        if self.result_count <= 0 or self.target_result_min is None or self.target_result_max is None:
            return False
        return self.target_result_min <= self.result_count <= self.target_result_max

    @property
    def outreach_signal_count(self) -> int:
        """Settled SAVE-family outcomes, with physical saves as a legacy proxy."""
        return max(self.full_outreach, self.saves)

    @property
    def settled_positive_count(self) -> int:
        return self.outreach_signal_count + self.full_review

    @property
    def effective_full_reviewed(self) -> int:
        return max(
            self.full_reviewed,
            self.outreach_signal_count + self.full_review + self.full_reject,
        )

    @property
    def all_reviewed_rejected(self) -> bool:
        return (
            self.effective_full_reviewed > 0
            and self.settled_positive_count == 0
            and self.full_reject >= self.effective_full_reviewed
        )

    def classify_result_window(self) -> str:
        """Deterministic health classification from observed metrics."""
        if self.result_count <= 0:
            return "too_narrow"
        if self.target_result_min is None or self.target_result_max is None:
            return "healthy" if self.result_count > 0 else "too_narrow"
        if self.result_count < self.target_result_min:
            return "too_narrow"
        if self.result_count > self.target_result_max:
            if self.pages_reviewed > 0 and self.settled_positive_count == 0:
                return "noisy"
            return "too_broad"
        if self.within_target_window() and self.pages_reviewed >= 1:
            if self.settled_positive_count == 0 and self.candidates > 3:
                return "misleading"
        return "healthy"

    def score(self) -> float:
        score = float(
            self.outreach_signal_count * 10
            + self.full_review * 2
            - self.full_reject
            - self.facial_no
        )
        if self.all_reviewed_rejected:
            return score
        if self.within_target_window():
            score += 5.0
        elif self.result_count > 0 and self.target_result_min is not None and self.target_result_max is not None:
            midpoint = (self.target_result_min + self.target_result_max) / 2
            distance = abs(self.result_count - midpoint) / max(midpoint, 1)
            score += max(0.0, 3.0 - distance * 3.0)
        return score


@dataclass
class LinkedInExperimentState:
    root_string_id: int
    intent: LinkedInSearchIntent
    mode: str = "recon"
    active_variant_id: str = "root"
    committed_variant_id: str | None = None
    planned_variant_ids: list[str] = field(default_factory=list)
    experiment_round: int = 0
    lane_id: str = ""
    mutations_used: int = 0
    consecutive_mutations: int = 0
    pages_since_last_mutation: int = 0
    executed_sibling_count: int = 0
    family_pages_reviewed_total: int = 0
    family_candidates_total: int = 0
    family_duplicates_total: int = 0
    family_signal_total: int = 0
    family_saves_total: int = 0
    family_reviewed_total: int = 0
    family_outreach_total: int = 0
    family_review_total: int = 0
    family_reject_total: int = 0
    precommit_recovery_attempts_used: int = 0
    committed_pages_reviewed: int = 0
    committed_zero_signal_streak: int = 0
    early_signal_snapshot: LinkedInVariantSnapshot | None = None
    recent_noise_snapshot: LinkedInVariantSnapshot | None = None
    drift_attempt_count: int = 0
    # Phase 2 hop 4 (slice E): per-lane count of structured demote-and-proceed
    # events (a structured control dropped/failed at apply while the keyword landed,
    # so the lane fell back to keyword-led). Drives the deterministic circuit-breaker
    # that stops the planners offering the promote/structured lever once it reaches
    # config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT.
    structured_demotions: int = 0
    pending_drift_variant_id: str | None = None
    pending_drift_parent_variant_id: str | None = None
    pending_drift_started_at: str = ""
    last_drift_refinement_summary: dict[str, Any] = field(default_factory=dict)
    variants: dict[str, LinkedInSearchVariant] = field(default_factory=dict)
    last_page_insights: LinkedInPageInsights | None = None
    last_variant_decision: dict[str, Any] = field(default_factory=dict)
    # Root-level shadow checkpoint. Generic causality/frontier payloads are
    # authored by the orchestrator so this state layer need not know queue
    # disposition semantics.
    allocator_last_observation: PageObservation | None = None
    allocator_last_verdict: dict[str, Any] = field(default_factory=dict)
    allocator_shadow_diverged: bool = False
    allocator_causality: dict[str, Any] = field(default_factory=dict)
    allocator_frontier_expectation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "root" not in self.variants:
            self.variants["root"] = LinkedInSearchVariant(
                variant_id="root",
                parent_variant_id=None,
                root_string_id=self.root_string_id,
                boolean=self.intent.root_boolean,
                variant_kind="original",
                status="active",
                experiment_round=0,
            )
        if (
            self.allocator_last_observation is not None
            and (
                self.allocator_last_observation.root_string_id
                != self.root_string_id
                or self.allocator_last_observation.variant_id not in self.variants
            )
        ):
            self.allocator_last_observation = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_string_id": self.root_string_id,
            "intent": self.intent.to_dict(),
            "mode": self.mode,
            "active_variant_id": self.active_variant_id,
            "committed_variant_id": self.committed_variant_id,
            "planned_variant_ids": list(self.planned_variant_ids),
            "experiment_round": self.experiment_round,
            "mutations_used": self.mutations_used,
            "consecutive_mutations": self.consecutive_mutations,
            "pages_since_last_mutation": self.pages_since_last_mutation,
            "executed_sibling_count": self.executed_sibling_count,
            "family_pages_reviewed_total": self.family_pages_reviewed_total,
            "family_candidates_total": self.family_candidates_total,
            "family_duplicates_total": self.family_duplicates_total,
            "family_signal_total": self.family_signal_total,
            "family_saves_total": self.family_saves_total,
            "family_reviewed_total": self.family_reviewed_total,
            "family_outreach_total": self.family_outreach_total,
            "family_review_total": self.family_review_total,
            "family_reject_total": self.family_reject_total,
            "precommit_recovery_attempts_used": self.precommit_recovery_attempts_used,
            "committed_pages_reviewed": self.committed_pages_reviewed,
            "committed_zero_signal_streak": self.committed_zero_signal_streak,
            "early_signal_snapshot": self.early_signal_snapshot.to_dict() if self.early_signal_snapshot else None,
            "recent_noise_snapshot": self.recent_noise_snapshot.to_dict() if self.recent_noise_snapshot else None,
            "drift_attempt_count": self.drift_attempt_count,
            "structured_demotions": self.structured_demotions,
            "pending_drift_variant_id": self.pending_drift_variant_id,
            "pending_drift_parent_variant_id": self.pending_drift_parent_variant_id,
            "pending_drift_started_at": self.pending_drift_started_at,
            "last_drift_refinement_summary": dict(self.last_drift_refinement_summary),
            "variants": {key: variant.to_dict() for key, variant in self.variants.items()},
            "last_page_insights": self.last_page_insights.to_dict() if self.last_page_insights else None,
            **(
                {
                    "allocator_last_observation": (
                        self.allocator_last_observation.to_dict()
                    )
                }
                if self.allocator_last_observation is not None
                else {}
            ),
            **(
                {"allocator_last_verdict": dict(self.allocator_last_verdict)}
                if self.allocator_last_verdict
                else {}
            ),
            **(
                {"allocator_shadow_diverged": True}
                if self.allocator_shadow_diverged
                else {}
            ),
            **(
                {"allocator_causality": dict(self.allocator_causality)}
                if self.allocator_causality
                else {}
            ),
            **(
                {
                    "allocator_frontier_expectation": dict(
                        self.allocator_frontier_expectation
                    )
                }
                if self.allocator_frontier_expectation
                else {}
            ),
            **({"lane_id": self.lane_id} if self.lane_id else {}),
            **({"last_variant_decision": dict(self.last_variant_decision)} if self.last_variant_decision else {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LinkedInExperimentState | None":
        if not payload:
            return None
        variants = {
            key: LinkedInSearchVariant.from_dict(value)
            for key, value in dict(payload.get("variants", {})).items()
        }
        family_saves_total = int(payload.get("family_saves_total", 0) or 0)
        family_outreach_total = int(
            payload.get("family_outreach_total", family_saves_total) or 0
        )
        family_review_total = int(payload.get("family_review_total", 0) or 0)
        family_reject_total = int(
            payload.get(
                "family_reject_total",
                sum(variant.full_reject for variant in variants.values()),
            )
            or 0
        )
        family_reviewed_total = int(
            payload.get(
                "family_reviewed_total",
                family_outreach_total + family_review_total + family_reject_total,
            )
            or 0
        )
        family_reviewed_total = max(
            family_reviewed_total,
            family_outreach_total + family_review_total + family_reject_total,
        )
        # Derived deliberately: legacy ``family_signal_total`` included raw
        # facial positives and rejects, so carrying it forward would silently
        # restore the behavior this contract removes.
        family_signal_total = family_outreach_total + family_review_total
        state = cls(
            root_string_id=int(payload.get("root_string_id", 0)),
            intent=LinkedInSearchIntent.from_dict(payload.get("intent")),
            mode=str(payload.get("mode", "recon")),
            active_variant_id=str(payload.get("active_variant_id", "root")),
            committed_variant_id=payload.get("committed_variant_id"),
            planned_variant_ids=list(payload.get("planned_variant_ids", [])),
            experiment_round=int(payload.get("experiment_round", 0)),
            mutations_used=int(payload.get("mutations_used", 0)),
            consecutive_mutations=int(payload.get("consecutive_mutations", 0)),
            pages_since_last_mutation=int(payload.get("pages_since_last_mutation", 0)),
            executed_sibling_count=int(payload.get("executed_sibling_count", 0)),
            family_pages_reviewed_total=int(payload.get("family_pages_reviewed_total", 0)),
            family_candidates_total=int(payload.get("family_candidates_total", 0)),
            family_duplicates_total=int(payload.get("family_duplicates_total", 0)),
            family_signal_total=family_signal_total,
            family_saves_total=family_saves_total,
            family_reviewed_total=family_reviewed_total,
            family_outreach_total=family_outreach_total,
            family_review_total=family_review_total,
            family_reject_total=family_reject_total,
            precommit_recovery_attempts_used=int(payload.get("precommit_recovery_attempts_used", 0)),
            committed_pages_reviewed=int(payload.get("committed_pages_reviewed", 0)),
            committed_zero_signal_streak=int(payload.get("committed_zero_signal_streak", 0)),
            early_signal_snapshot=LinkedInVariantSnapshot.from_dict(payload.get("early_signal_snapshot")),
            recent_noise_snapshot=LinkedInVariantSnapshot.from_dict(payload.get("recent_noise_snapshot")),
            drift_attempt_count=int(payload.get("drift_attempt_count", 0)),
            structured_demotions=int(payload.get("structured_demotions", 0)),
            pending_drift_variant_id=payload.get("pending_drift_variant_id"),
            pending_drift_parent_variant_id=payload.get("pending_drift_parent_variant_id"),
            pending_drift_started_at=str(payload.get("pending_drift_started_at", "")),
            last_drift_refinement_summary=dict(payload.get("last_drift_refinement_summary", {})),
            variants=variants,
            lane_id=str(payload.get("lane_id", "")),
            last_page_insights=LinkedInPageInsights.from_dict(payload.get("last_page_insights")),
            last_variant_decision=dict(payload.get("last_variant_decision", {})),
            allocator_last_observation=_shadow_observation(
                payload.get("allocator_last_observation")
            ),
            allocator_last_verdict=_shadow_dict(
                payload.get("allocator_last_verdict")
            ),
            allocator_shadow_diverged=(
                payload.get("allocator_shadow_diverged", False)
                if isinstance(payload.get("allocator_shadow_diverged", False), bool)
                else False
            ),
            allocator_causality=_shadow_dict(payload.get("allocator_causality")),
            allocator_frontier_expectation=_shadow_dict(
                payload.get("allocator_frontier_expectation")
            ),
        )
        state.__post_init__()
        return state

    @property
    def root_variant(self) -> LinkedInSearchVariant:
        return self.variants["root"]

    @property
    def active_variant(self) -> LinkedInSearchVariant:
        return self.variants[self.active_variant_id]

    @property
    def committed_variant(self) -> LinkedInSearchVariant | None:
        if not self.committed_variant_id:
            return None
        return self.variants.get(self.committed_variant_id)

    def current_boolean(self) -> str:
        if self.active_variant_id in self.variants:
            return self.active_variant.boolean
        return self.root_variant.boolean

    def active_allocator_page_cursor(self) -> int:
        """Return the first incomplete page for the active search variant."""
        active = self.variants.get(self.active_variant_id)
        return int(active.allocator_page_cursor or 0) if active is not None else 0

    def set_active_allocator_page_cursor(self, page_num: int) -> None:
        """Set the first incomplete page on the active search variant."""
        cursor = max(0, int(page_num or 0))
        active = self.variants.get(self.active_variant_id)
        if active is not None:
            active.allocator_page_cursor = cursor

    def root_has_valid_probe(self) -> bool:
        """Whether any rewrite under this root has produced teaching currency."""

        return any(
            variant.allocator_valid_page_count > 0
            for variant in self.variants.values()
        )

    def legacy_unobserved_pages(self, variant_id: str | None = None) -> int:
        """Legacy completed pages not represented by TUR-14 observations."""

        variant = self.variants.get(variant_id or self.active_variant_id)
        if variant is None:
            return 0
        return max(
            0,
            variant.pages_reviewed
            - variant.allocator_completed_observation_count,
        )

    def record_allocator_observation(self, observation: PageObservation) -> None:
        """Record one canonically completed page against its exact variant."""

        if observation.root_string_id != self.root_string_id:
            raise ValueError("allocator observation belongs to another root")
        variant = self.variants.get(observation.variant_id)
        if variant is None:
            raise ValueError("allocator observation belongs to an unknown variant")

        self.allocator_last_observation = observation
        variant.allocator_completed_observation_count += 1
        if not observation.teaches_policy:
            return
        variant.allocator_valid_page_count += 1
        variant.allocator_observations = [
            *variant.allocator_observations,
            observation,
        ][-2:]

    def compat_refinement_stack(self) -> list[str]:
        lineage = self.variant_lineage(self.active_variant_id)
        return [variant.boolean for variant in lineage[:-1]]

    def compat_phase(self) -> str:
        return "paginate" if self.mode in {"paginate", "drift"} else "scout"

    def apply_shadow(self, search_string: SearchString) -> None:
        search_string.boolean = self.current_boolean()
        search_string.original_boolean = self.intent.root_boolean
        search_string.refinement_stack = self.compat_refinement_stack()
        search_string.phase = self.compat_phase()
        # Phase 2 hop 4 (slice G): persist the active variant's surface AND its
        # CURRENT structured_filters onto the compat SearchString. SearchString has
        # no LinkedInSearchVariant, so a worker-death / cross-process resume that
        # rebuilds via bootstrap_experiment_state (NOT the in-memory from_dict) reads
        # ONLY these compat fields. Without them the resumed variant degrades to
        # surface="" (the slice-D keyword suppression stops firing, so a keyword can
        # re-enter a structured_only lane) and to the PRODUCER-TIME filters (a mid-run
        # promote is lost).
        active = self.active_variant
        search_string.surface = active.surface
        # The CURRENT-filters write UNIFIES and SUBSUMES the slice-C special-case
        # demote-clear into one rule: persist the active variant's OWN filters whenever
        # it carries them (a PROMOTE -> the promoted filters survive resume) OR it is a
        # deliberate boolean demotion (-> empty filters == the slice-C clear, so
        # bootstrap seeds nothing and the demote-no-reseed behavior is preserved).
        #
        # The one case that must NOT write is a plain keyword refinement
        # (precision/recall) that never carried its own filters while the lane DID:
        # writing its empty filters would WIPE the lane geography off the SearchString,
        # and the slice-B legacy-variant re-seed (bootstrap) would then find nothing to
        # restore — a mutated hybrid lane would resume as a bare keyword search. Leaving
        # the checkpointed lane filters untouched in that case keeps slice B intact; a
        # demote is still caught because is_deliberate_boolean_demotion fires on it.
        #
        # A deliberate demote writes the BARE {} (the slice-C clear marker) rather than
        # a fully-keyed all-empty dict, so the canonical "no filters" sentinel the rest
        # of the system keys on (the seed early-return's `if not structured_filters`,
        # the checkpoint's empty-dict default) is byte-preserved.
        if not active.structured_filters.is_empty():
            search_string.structured_filters = active.structured_filters.to_dict()
        elif is_deliberate_boolean_demotion(active):
            search_string.structured_filters = {}

    def variant_lineage(self, variant_id: str | None = None) -> list[LinkedInSearchVariant]:
        variant_id = variant_id or self.active_variant_id
        lineage: list[LinkedInSearchVariant] = []
        seen: set[str] = set()
        current = self.variants.get(variant_id)
        while current and current.variant_id not in seen:
            lineage.append(current)
            seen.add(current.variant_id)
            current = self.variants.get(current.parent_variant_id or "")
        lineage.reverse()
        return lineage or [self.root_variant]

    def note_page_review(self) -> None:
        self.consecutive_mutations = 0
        self.pages_since_last_mutation += 1
        if self.pending_drift_variant_id and self.pending_drift_variant_id == self.active_variant_id:
            self.clear_pending_drift()

    def begin_experiment_round(self, variants: list[LinkedInSearchVariant]) -> None:
        self.experiment_round += 1
        self.mode = "experiment"
        self.executed_sibling_count = 0
        self.planned_variant_ids = []
        parent_id = self.active_variant_id
        for index, variant in enumerate(variants[:3], start=1):
            variant.parent_variant_id = variant.parent_variant_id or parent_id
            variant.root_string_id = self.root_string_id
            variant.experiment_round = self.experiment_round
            variant.status = "planned"
            if not variant.variant_id:
                variant.variant_id = f"round-{self.experiment_round}-{index}"
            self.variants[variant.variant_id] = variant
            self.planned_variant_ids.append(variant.variant_id)

    def next_planned_variant(self) -> LinkedInSearchVariant | None:
        for variant_id in self.planned_variant_ids:
            variant = self.variants.get(variant_id)
            if variant and variant.status == "planned":
                return variant
        return None

    def activate_variant(self, variant_id: str) -> LinkedInSearchVariant:
        current_mode = self.mode
        if self.active_variant_id in self.variants and self.variants[self.active_variant_id].status == "active":
            self.variants[self.active_variant_id].status = "explored"
        variant = self.variants[variant_id]
        if variant_id != "root" and self.mode == "experiment":
            variant.status = "probing"
        else:
            variant.status = "active"
        self.active_variant_id = variant_id
        self.mutations_used += 1
        self.consecutive_mutations += 1
        self.pages_since_last_mutation = 0
        if variant_id in self.planned_variant_ids:
            self.executed_sibling_count += 1
            if current_mode in {"recon", "experiment"} and self.committed_variant_id is None:
                self.precommit_recovery_attempts_used += 1
        return variant

    def commit_variant(self, variant_id: str | None = None) -> LinkedInSearchVariant:
        variant_id = variant_id or self.active_variant_id
        variant = self.variants[variant_id]
        preserving_drift_summary = self.mode == "drift"
        variant.status = "committed"
        self.committed_variant_id = variant_id
        self.active_variant_id = variant_id
        self.mode = "paginate"
        self.planned_variant_ids = []
        self.executed_sibling_count = 0
        self.committed_pages_reviewed = 0
        self.committed_zero_signal_streak = 0
        self.early_signal_snapshot = None
        self.recent_noise_snapshot = None
        self.drift_attempt_count = 0
        self.pending_drift_variant_id = None
        self.pending_drift_parent_variant_id = None
        self.pending_drift_started_at = ""
        if not preserving_drift_summary:
            self.last_drift_refinement_summary = {}
        return variant

    def record_variant_metrics(
        self,
        *,
        variant_id: str | None = None,
        page_num: int,
        result_count: int,
        page_stats: dict[str, Any],
        page_insights: LinkedInPageInsights | None = None,
    ) -> None:
        variant = self.variants[variant_id or self.active_variant_id]
        variant.result_count = result_count
        variant.pages_reviewed = max(variant.pages_reviewed, page_num)
        variant.candidates += int(page_stats.get("candidates", 0))
        variant.duplicates += int(page_stats.get("duplicates", 0))
        variant.saves += int(page_stats.get("saves", 0))
        variant.rejects += int(page_stats.get("rejects", 0))
        full_reviewed, full_outreach, full_review, full_reject = _full_outcome_metrics(page_stats)
        variant.full_reviewed += full_reviewed
        variant.full_outreach += full_outreach
        variant.full_review += full_review
        variant.full_reject += full_reject
        variant.facial_yes += int(page_stats.get("facial_yes", 0))
        variant.facial_borderline += int(page_stats.get("facial_borderline", 0))
        variant.facial_no += int(page_stats.get("facial_no", 0))
        variant.last_page_insights = page_insights
        variant.probe_pages_used = max(variant.probe_pages_used, variant.pages_reviewed)
        self.last_page_insights = page_insights
        if variant.status == "planned":
            variant.status = "probing" if self.mode == "experiment" else "explored"

    def record_family_page_metrics(
        self,
        *,
        page_num: int,
        result_count: int,
        page_stats: dict[str, int],
        page_insights: LinkedInPageInsights,
    ) -> None:
        self.family_pages_reviewed_total += 1
        self.family_candidates_total += int(page_stats.get("candidates", 0))
        self.family_duplicates_total += int(page_stats.get("duplicates", 0))
        full_reviewed, full_outreach, full_review, full_reject = _full_outcome_metrics(page_stats)
        page_signal = full_outreach + full_review
        self.family_signal_total += page_signal
        self.family_saves_total += int(page_stats.get("saves", 0))
        self.family_reviewed_total += full_reviewed
        self.family_outreach_total += full_outreach
        self.family_review_total += full_review
        self.family_reject_total += full_reject

        is_committed_variant_page = (
            self.committed_variant_id is not None and self.active_variant_id == self.committed_variant_id
        )
        if is_committed_variant_page and self.early_signal_snapshot is None:
            if page_signal > 0:
                self.early_signal_snapshot = LinkedInVariantSnapshot.from_page(
                    page_num=page_num,
                    result_count=result_count,
                    page_insights=page_insights,
                    page_stats=page_stats,
                )
        if is_committed_variant_page:
            self.committed_pages_reviewed += 1
            no_signal = page_signal == 0
            noisy_page = bool(page_insights.noise_anchors) or page_insights.glance_action == "reformulate"
            if no_signal:
                self.committed_zero_signal_streak += 1
            else:
                self.committed_zero_signal_streak = 0
                if self.last_drift_refinement_summary.get("outcome") == "not_rescued":
                    self.last_drift_refinement_summary = {
                        **self.last_drift_refinement_summary,
                        "outcome": "signal_returned",
                    }
            if no_signal and noisy_page:
                self.recent_noise_snapshot = LinkedInVariantSnapshot.from_page(
                    page_num=page_num,
                    result_count=result_count,
                    page_insights=page_insights,
                    page_stats=page_stats,
                )

    def real_signal_seen(self) -> bool:
        return self.family_outreach_total > 0

    def mark_pending_drift(
        self,
        *,
        variant_id: str,
        parent_variant_id: str | None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        self.mode = "drift"
        self.drift_attempt_count += 1
        self.pending_drift_variant_id = variant_id
        self.pending_drift_parent_variant_id = parent_variant_id
        self.pending_drift_started_at = datetime.now(timezone.utc).isoformat()
        if summary is not None:
            self.last_drift_refinement_summary = dict(summary)

    def clear_pending_drift(self, summary: dict[str, Any] | None = None) -> None:
        self.pending_drift_variant_id = None
        self.pending_drift_parent_variant_id = None
        self.pending_drift_started_at = ""
        if summary is not None:
            self.last_drift_refinement_summary = dict(summary)

    def rollback_pending_drift(self) -> None:
        if self.drift_attempt_count > 0:
            self.drift_attempt_count -= 1
        self.pending_drift_variant_id = None
        self.pending_drift_parent_variant_id = None
        self.pending_drift_started_at = ""
        if self.mode == "drift":
            self.mode = "paginate" if self.committed_variant_id else "recon"

    def resume_committed_after_failed_drift(self) -> None:
        if self.active_variant_id in self.variants:
            self.variants[self.active_variant_id].status = "explored"
        if self.committed_variant_id:
            self.active_variant_id = self.committed_variant_id
        self.pending_drift_variant_id = None
        self.pending_drift_parent_variant_id = None
        self.pending_drift_started_at = ""
        self.mode = "paginate" if self.committed_variant_id else "recon"
        self.pages_since_last_mutation = 0

    def best_variant(self) -> LinkedInSearchVariant:
        candidates = [
            variant
            for variant in self.variants.values()
            if variant.pages_reviewed > 0 or variant.result_count > 0 or variant.variant_id == self.active_variant_id
        ]
        return max(candidates, key=lambda variant: variant.score(), default=self.active_variant)

    def metrics_summary(self) -> dict[str, Any]:
        active_variant = self.active_variant
        return {
            "mode": self.mode,
            "active_variant_id": self.active_variant_id,
            "committed_variant_id": self.committed_variant_id,
            "experiment_round": self.experiment_round,
            "mutations_used": self.mutations_used,
            "drift_attempt_count": self.drift_attempt_count,
            "structured_demotions": self.structured_demotions,
            "pending_drift_variant_id": self.pending_drift_variant_id,
            "pending_drift_parent_variant_id": self.pending_drift_parent_variant_id,
            "pending_drift_started_at": self.pending_drift_started_at,
            "family_pages_reviewed_total": self.family_pages_reviewed_total,
            "family_candidates_total": self.family_candidates_total,
            "family_duplicates_total": self.family_duplicates_total,
            "family_signal_total": self.family_signal_total,
            "family_saves_total": self.family_saves_total,
            "family_reviewed_total": self.family_reviewed_total,
            "family_outreach_total": self.family_outreach_total,
            "family_review_total": self.family_review_total,
            "family_reject_total": self.family_reject_total,
            "precommit_recovery_attempts_used": self.precommit_recovery_attempts_used,
            "committed_pages_reviewed": self.committed_pages_reviewed,
            "committed_zero_signal_streak": self.committed_zero_signal_streak,
            "active_variant_page": active_variant.pages_reviewed,
            "active_variant_pages_reviewed": active_variant.pages_reviewed,
            "active_variant_result_count": active_variant.result_count,
            "active_variant_signal": active_variant.settled_positive_count,
            "active_variant_saves": active_variant.saves,
            "active_variant_full_reviewed": active_variant.full_reviewed,
            "active_variant_full_outreach": active_variant.full_outreach,
            "active_variant_full_review": active_variant.full_review,
            "active_variant_full_reject": active_variant.full_reject,
            "executed_sibling_count": self.executed_sibling_count,
            "early_signal_snapshot": self.early_signal_snapshot.to_dict() if self.early_signal_snapshot else None,
            "recent_noise_snapshot": self.recent_noise_snapshot.to_dict() if self.recent_noise_snapshot else None,
            "family_outcome_summary": {
                "root_string_id": self.root_string_id,
                "committed_variant_id": self.committed_variant_id,
                "family_pages_reviewed_total": self.family_pages_reviewed_total,
                "family_signal_total": self.family_signal_total,
                "family_saves_total": self.family_saves_total,
                "family_reviewed_total": self.family_reviewed_total,
                "family_outreach_total": self.family_outreach_total,
                "family_review_total": self.family_review_total,
                "family_reject_total": self.family_reject_total,
            },
            "drift_rescue_summary": dict(self.last_drift_refinement_summary),
            "variants": {
                key: {
                    "variant_kind": variant.variant_kind,
                    "status": variant.status,
                    "result_count": variant.result_count,
                    "pages_reviewed": variant.pages_reviewed,
                    "saves": variant.saves,
                    "full_reviewed": variant.full_reviewed,
                    "full_outreach": variant.full_outreach,
                    "full_review": variant.full_review,
                    "full_reject": variant.full_reject,
                    "facial_yes": variant.facial_yes,
                    "facial_borderline": variant.facial_borderline,
                    "facial_no": variant.facial_no,
                    "score": round(variant.score(), 2),
                }
                for key, variant in self.variants.items()
            },
        }


def _quoted_or_group(terms: list[str]) -> str:
    return " OR ".join(f'"{term}"' for term in terms if term)


def _append_required_terms(boolean: str, terms: list[str]) -> str:
    group = _quoted_or_group(terms[:3])
    if not group:
        return boolean
    return f"({boolean}) AND ({group})" if boolean else f"({group})"


def _append_not_terms(boolean: str, terms: list[str]) -> str:
    group = _quoted_or_group(terms[:3])
    if not group:
        return boolean
    return f"({boolean}) NOT ({group})" if boolean else f"NOT ({group})"


def _broaden_boolean(boolean: str) -> str:
    if " NOT " in boolean:
        return boolean.rsplit(" NOT ", 1)[0].strip()
    parts = boolean.split(" AND ")
    if len(parts) > 1:
        return " AND ".join(parts[:-1]).strip()
    return boolean


def is_deliberate_boolean_demotion(variant: LinkedInSearchVariant) -> bool:
    """Phase 2 hop 4 (slice C, part 5): True when this variant deliberately demoted
    its structured filters back to a keyword-only search.

    The discriminator survives a to_dict/from_dict round-trip: surface=="boolean"
    (chose keyword entry) AND variant_kind=="structured_filter" (ran the structured
    planner and acted on it). A plain keyword variant (precision/recall/...) is
    surface=="boolean" but a non-structured kind, so it still seeds; a legacy or
    never-touched variant has surface=="" and seeds too. This lets the seeding
    carryover skip a variant that chose to drop its filters without clobbering one
    that simply never had them.
    """
    return variant.surface == "boolean" and variant.variant_kind == "structured_filter"


def _copy_filters(filters: LinkedInStructuredFilters) -> LinkedInStructuredFilters:
    return LinkedInStructuredFilters.from_dict(filters.to_dict())


def _drop_one_filter(filters: LinkedInStructuredFilters) -> bool:
    for values in (filters.assessments, filters.skills, filters.companies, filters.titles):
        if values:
            values.pop()
            return True
    for bucket in (filters.advanced_filters, filters.sidebar_filters):
        for key in list(bucket.keys()):
            value = bucket.get(key)
            if isinstance(value, list) and value:
                value.pop()
                if not value:
                    bucket.pop(key, None)
                return True
            if value:
                bucket.pop(key, None)
                return True
    return False


def _signal_terms(variant: LinkedInSearchVariant) -> list[str]:
    insights = variant.last_page_insights
    if not insights:
        return []
    terms = [str(term).strip() for term in insights.signal_anchors if str(term).strip()]
    for cluster in insights.title_clusters:
        if isinstance(cluster, dict) and int(cluster.get("signal_count", 0) or 0) > 0:
            raw = cluster.get("title") or cluster.get("label") or cluster.get("name")
            if raw and str(raw).strip():
                terms.append(str(raw).strip())
    return list(dict.fromkeys(terms))


def _noise_terms(variant: LinkedInSearchVariant) -> list[str]:
    insights = variant.last_page_insights
    if not insights:
        return []
    terms = [str(term).strip() for term in insights.noise_anchors if str(term).strip()]
    return list(dict.fromkeys(terms))


def spawn_rescue_variant_from_hint(
    parent: LinkedInSearchVariant,
    *,
    hint: dict[str, Any] | None,
    root_string_id: int,
) -> LinkedInSearchVariant | None:
    """Create a bounded follow-up variant from a lifecycle rescue/split hint."""
    kind = "rescue"
    action = ""
    if hint:
        kind = str(hint.get("variant_kind", "rescue") or "rescue")
        action = str(hint.get("action", "") or "")
    suffix = f"{kind}-{action}".strip("-") or kind
    next_boolean = parent.boolean
    next_filters = _copy_filters(parent.structured_filters)

    if kind in {"recall", "broadening"} or action == "broaden":
        dropped_filter = _drop_one_filter(next_filters)
        next_boolean = _broaden_boolean(next_boolean)
        if not dropped_filter and next_boolean == parent.boolean:
            return None
    elif kind in {"precision", "noise_exclusion"} or action == "narrow":
        terms = _noise_terms(parent)
        next_boolean = _append_not_terms(next_boolean, terms)
        if next_boolean == parent.boolean:
            next_boolean = _append_required_terms(next_boolean, _signal_terms(parent))
        if next_boolean == parent.boolean:
            return None
    elif kind == "keyword_focus":
        next_boolean = _append_required_terms(next_boolean, _signal_terms(parent))
        if next_boolean == parent.boolean:
            return None
    elif kind == "structured_filter":
        if next_filters.is_empty():
            return None
    elif kind == "rescue":
        pass
    else:
        return None

    # Slice F: the rescue spawner is the fifth window-write site. The four build sites
    # (orchestrator._plan_variant_experiments / _plan_drift_refinement, the forced-narrow
    # fallback, bootstrap_experiment_state) scale_window_for_surface against the CHILD's
    # resolved surface; this site instead INHERITS the parent window. Inheriting raw is
    # correct only when posture is unchanged. Resolve the child's surface from its actual
    # (boolean, filters) and re-project the window across any posture flip so a broaden
    # that drops the last filter (filter-led parent -> runnable keyword child) un-scales
    # back to keyword bounds — closing the directional inverse of the dead-end slice F
    # fixed, where the gate would mis-read a healthy keyword count as too_broad.
    next_surface = _surface_for_rescue_child(next_boolean, next_filters)
    next_min, next_max = reproject_rescue_window(
        parent.target_result_min,
        parent.target_result_max,
        parent_surface=parent.surface,
        parent_filters=parent.structured_filters,
        child_surface=next_surface,
        child_filters=next_filters,
    )

    return LinkedInSearchVariant(
        variant_id=f"{parent.variant_id}-{suffix}-{parent.probe_pages_used + 1}",
        parent_variant_id=parent.variant_id,
        root_string_id=root_string_id,
        boolean=next_boolean,
        variant_kind=kind,
        hypothesis=f"Lifecycle {suffix} spawned from {parent.variant_id}",
        target_result_min=next_min,
        target_result_max=next_max,
        status="planned",
        lane_id=parent.lane_id,
        lifecycle_reason=f"spawned_from_{suffix}",
        structured_filters=next_filters,
        surface=next_surface,
        probe_page_budget=parent.probe_page_budget,
    )


def seed_structured_filters_onto_variants(
    structured_filters: dict[str, Any],
    variants: list[LinkedInSearchVariant],
) -> None:
    """Phase 2 hop 3: seed runtime variants with the SearchString's structured filters.

    Hybrid lanes carry their structured filters on the SearchString (hop 2). The
    runtime variants built per string (orchestrator._plan_variant_experiments /
    _plan_drift_refinement) start with empty filters, so apply_variant would take the
    keyword-only path. Seeding lets a hybrid lane's variant carry the title/company
    controls into apply_variant — pairing with the lane compiler forcing
    acquisition_mode='linkedin_hybrid' whenever filters are present.

    Only seeds variants whose own filters are empty, so a variant that already
    carries its own filters is never clobbered. No-op for a boolean lane
    (empty dict) — the live keyword path is untouched.

    Phase 2 hop 4 (slice C, part 5a): a variant that DELIBERATELY demoted its
    structured filters to a keyword-only search (is_deliberate_boolean_demotion) is
    also skipped — re-seeding the lane's filters would silently undo the demote the
    adaptive loop chose. "Never had filters" (seed it) is distinguished from
    "deliberately demoted to boolean" (do NOT re-seed) via the surface marker.
    """
    if not structured_filters:
        return
    for variant in variants:
        if is_deliberate_boolean_demotion(variant):
            continue
        if variant.structured_filters.is_empty():
            variant.structured_filters = LinkedInStructuredFilters.from_dict(structured_filters)


def compile_lane_variant_to_linkedin(
    lane_variant: Any,
    *,
    root_string_id: int = 0,
) -> LinkedInSearchVariant:
    """Map a shared LaneVariant to a LinkedIn-specific execution variant."""
    structured = LinkedInStructuredFilters()
    controls = getattr(lane_variant, "structured_controls", None) or {}
    if controls:
        structured = LinkedInStructuredFilters(
            titles=list(controls.get("titles", [])),
            companies=list(controls.get("companies", [])),
            sidebar_filters=dict(controls.get("sidebar_filters", {})),
            advanced_filters=dict(controls.get("advanced_filters", {})),
        )
    return LinkedInSearchVariant(
        variant_id=lane_variant.variant_id,
        parent_variant_id=None,
        root_string_id=root_string_id,
        boolean=getattr(lane_variant, "boolean_intent", "") or "",
        variant_kind=getattr(lane_variant, "variant_kind", "original") or "original",
        hypothesis=getattr(lane_variant, "hypothesis", "") or "",
        target_result_min=getattr(lane_variant, "target_result_min", None),
        target_result_max=getattr(lane_variant, "target_result_max", None),
        status=getattr(lane_variant, "status", "planned") or "planned",
        structured_filters=structured,
        lane_id=getattr(lane_variant, "lane_id", "") or "",
        lifecycle_reason=getattr(lane_variant, "reason", "") or "",
        probe_page_budget=int((getattr(lane_variant, "probe_budget", None) or {}).get("page_limit", 1)),
    )


def bootstrap_experiment_state(search_string: SearchString) -> LinkedInExperimentState:
    """Bootstrap experiment state from compatibility-era SearchString fields."""
    root_boolean = search_string.original_boolean or search_string.boolean
    # Phase 2 hop 4 (slice B, part 1): carry the lane's compiled structured filters
    # (set on the checkpointed SearchString by slice A's producer path) onto the
    # intent so reset_experiment_state inherits them (:1043), and onto the root
    # variant so the opening structured-apply (orchestrator._process_string) and
    # cross-process resume drive off active.structured_filters. No-op on an
    # all-boolean lane (empty dict) — the all-boolean default is byte-preserved.
    structured_filters = LinkedInStructuredFilters.from_dict(search_string.structured_filters)
    intent = LinkedInSearchIntent(
        root_boolean=root_boolean,
        family_key=search_string.family_key,
        novelty_bucket=search_string.novelty_bucket,
        domain_lane=search_string.domain_lane,
        retrieval_recipe=dict(search_string.retrieval_recipe or {}),
        applied_hypothesis_ids=list(search_string.retrieval_hypothesis_ids or []),
        structured_filters=structured_filters,
    )
    state = LinkedInExperimentState(
        root_string_id=search_string.id,
        intent=intent,
        mode="paginate" if search_string.phase == "paginate" else "recon",
    )
    # Phase 2 hop 4 (slice G): reconstruct the active variant's surface from the
    # compat SearchString (apply_shadow persists active.surface there). Without this
    # the root mints with surface="" and a resumed structured_only lane loses the
    # slice-D include_keyword gate (the keyword re-enters); a resumed promote/demote
    # loses its surface-aware state. Stamp the root here, before the legacy chain may
    # repoint the active variant — the chain branch re-stamps the chosen active
    # variant below. No-op default "" on a keyword-led / legacy lane (byte-preserved).
    state.root_variant.surface = search_string.surface
    seed_structured_filters_onto_variants(
        dict(search_string.structured_filters or {}), [state.root_variant]
    )
    # Slice F: the root seed is a build site. A filter-led root (the lane stamped its
    # structured filters above) gets its target window scaled DOWN by the config
    # factor, exactly as the _plan_variant_experiments / _plan_drift_refinement build
    # sites do, so the keyword-tuned lifecycle gate does not abandon a legitimately
    # narrower structured root. The root window is None until a planner proposes one,
    # so this is a no-op today (None passes through) — and stays byte-identical on an
    # all-boolean lane (empty filters, default surface) — but it wires the seam so any
    # window set on a filter-led root is scaled at construction, never read by the gate.
    root = state.root_variant
    root.target_result_min, root.target_result_max = scale_window_for_surface(
        root.target_result_min,
        root.target_result_max,
        surface=root.surface,
        structured_filters=root.structured_filters,
    )

    chain = list(search_string.refinement_stack)
    if not chain:
        state.root_variant.boolean = root_boolean
        if state.mode == "paginate" and not state.committed_variant_id:
            # Same paginate-implies-committed invariant as the orchestrator's
            # direct-pagination entry: a chain-less rebuild in paginate mode
            # must commit the root or the zero-signal stop rule never arms
            # (this lossy-bootstrap path had already dropped the counters;
            # a committed root is strictly better than the dead state).
            state.commit_variant()
        state.apply_shadow(search_string)
        return state

    current_parent = "root"
    seen_boolean = root_boolean
    for index, boolean in enumerate(chain + [search_string.boolean], start=1):
        if not boolean or boolean == seen_boolean:
            continue
        variant_id = f"legacy-{index}"
        variant = LinkedInSearchVariant(
            variant_id=variant_id,
            parent_variant_id=current_parent,
            root_string_id=search_string.id,
            boolean=boolean,
            variant_kind="precision",
            status="committed" if boolean == search_string.boolean else "explored",
            experiment_round=0,
        )
        state.variants[variant_id] = variant
        current_parent = variant_id
        seen_boolean = boolean

    state.active_variant_id = current_parent
    state.committed_variant_id = current_parent
    state.mode = "paginate"
    # Phase 2 hop 4 (slice G): the legacy chain repointed the active variant at the
    # LAST legacy variant (minted with surface=""), so the root surface stamped above
    # does not reach the variant the resumed opening reads. Reconstruct the active
    # variant's surface from the compat SearchString here so a resumed structured_only
    # lane that committed a refinement still gates the keyword off (slice D), and a
    # resumed promote/demote reconstructs its surface-aware state on the active variant.
    # No-op default "" on a keyword-led / legacy lane (byte-preserved).
    state.active_variant.surface = search_string.surface
    # Phase 2 hop 4 (slice B, part 1): a lane that committed a refinement before a
    # crash checkpoints a non-empty refinement_stack (apply_shadow writes
    # search_string.refinement_stack = compat_refinement_stack(), :527), which drives
    # this legacy-chain branch and points active_variant_id at the LAST legacy variant
    # — NOT 'root'. The legacy variants are minted with empty structured_filters, so
    # seeding only the root above is not enough: _apply_opening_search reads the ACTIVE
    # variant's filters (orchestrator.py:~1834), finds them empty, and drops the
    # resumed hybrid lane to bare keyword entry, losing its geography. Seed every
    # legacy variant whose own filters are empty (seed_structured_filters_onto_variants
    # never clobbers a variant that already carries its own) so the active variant on
    # resume reconstructs the structured search. No-op on an all-boolean lane.
    seed_structured_filters_onto_variants(
        dict(search_string.structured_filters or {}),
        [
            variant
            for variant_id, variant in state.variants.items()
            if variant_id != "root"
        ],
    )
    state.apply_shadow(search_string)
    return state


def reset_experiment_state(
    search_string: SearchString,
    state: LinkedInExperimentState | None = None,
) -> LinkedInExperimentState:
    if state is None:
        state = bootstrap_experiment_state(search_string)
    intent = LinkedInSearchIntent(
        root_boolean=state.intent.root_boolean or search_string.original_boolean or search_string.boolean,
        family_key=state.intent.family_key or search_string.family_key,
        novelty_bucket=state.intent.novelty_bucket or search_string.novelty_bucket,
        domain_lane=state.intent.domain_lane or search_string.domain_lane,
        retrieval_recipe=state.intent.retrieval_recipe or dict(search_string.retrieval_recipe or {}),
        applied_hypothesis_ids=state.intent.applied_hypothesis_ids or list(search_string.retrieval_hypothesis_ids or []),
        structured_filters=state.intent.structured_filters,
    )
    reset_state = LinkedInExperimentState(root_string_id=search_string.id, intent=intent, mode="recon")
    # Phase 2 hop 4 (slice G): reconstruct the fresh root's surface from the compat
    # SearchString BEFORE apply_shadow, mirroring bootstrap_experiment_state:1250. The
    # restart/requeue path (shared/runtime_state/linkedin.py:547-550) is the SAME
    # surface-degradation class slice G closed on the worker-death bootstrap path, just
    # via reset_experiment_state: the __post_init__-minted root carries surface="", and
    # apply_shadow below writes that "" onto search_string.surface (:700) — wiping the
    # persisted structured_only posture while the intent still carries the filters
    # forward (:1342). The next bootstrap then reconstructs surface="" with non-empty
    # filters, and the slice-D include_keyword gate (active.surface != "structured_only")
    # re-admits the keyword the lane suppressed. Stamping the durable surface here keeps
    # the structured_only/hybrid posture coherent across a restart; a demote (surface
    # "boolean") and a keyword-led lane (default "") are byte-preserved — apply_shadow's
    # filter-write gate is unchanged because the fresh root's own filters stay empty and
    # its kind stays "original", so is_deliberate_boolean_demotion does not fire.
    reset_state.root_variant.surface = search_string.surface
    reset_state.apply_shadow(search_string)
    return reset_state
