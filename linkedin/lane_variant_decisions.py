"""Deterministic lane-variant lifecycle decisions — P7b.

Pure functions from observed metrics to commit/abandon/rescue/split/continue
decisions. No bandits, no LLM judgment. Deterministic thresholds with enough
persisted metrics to support future learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from linkedin.search_intelligence import LinkedInExperimentState, LinkedInSearchVariant
    from shared.sourcing_lanes import LaneProbe


@dataclass
class VariantDecisionInput:
    variant: "LinkedInSearchVariant"
    experiment_state: "LinkedInExperimentState"
    lane_probe: "LaneProbe | None" = None
    run_level_lane_metrics: dict[str, Any] | None = None


@dataclass
class VariantDecisionOutput:
    action: str
    reason: str
    next_variant_hint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action": self.action, "reason": self.reason}
        if self.next_variant_hint:
            d["next_variant_hint"] = self.next_variant_hint
        return d


def decide_variant_lifecycle(inp: VariantDecisionInput) -> VariantDecisionOutput:
    """Return a deterministic lifecycle decision for the given variant state.

    Decision priority:
    1. Budget exhausted → abandon
    2. Result-window health drives rescue/commit
    3. Within budget and unclassified → continue
    """
    variant = inp.variant
    health = variant.classify_result_window()
    variant.result_window_health = health

    probe_budget = variant.probe_page_budget
    probe_used = variant.probe_pages_used
    has_signal = variant.settled_positive_count > 0

    if variant.all_reviewed_rejected and probe_used >= probe_budget:
        return VariantDecisionOutput(
            action="abandon",
            reason="all_reviewed_rejected_without_full_profile_signal",
        )

    if health == "healthy" and has_signal:
        return VariantDecisionOutput(
            action="commit",
            reason="healthy_window_with_full_profile_signal",
        )

    if health == "healthy" and probe_used >= probe_budget:
        return VariantDecisionOutput(
            action="abandon",
            reason="healthy_window_budget_met_without_full_profile_signal",
        )

    if health == "too_narrow":
        if probe_used >= probe_budget:
            return VariantDecisionOutput(
                action="rescue",
                reason="too_narrow_budget_met",
                next_variant_hint={"variant_kind": "recall", "action": "broaden"},
            )
        return VariantDecisionOutput(
            action="continue",
            reason="too_narrow_within_budget",
        )

    if health == "too_broad":
        return VariantDecisionOutput(
            action="rescue",
            reason="too_broad",
            next_variant_hint={"variant_kind": "precision", "action": "narrow"},
        )

    if health == "noisy":
        return VariantDecisionOutput(
            action="rescue",
            reason="noisy_no_signal",
            next_variant_hint={"variant_kind": "noise_exclusion", "action": "narrow"},
        )

    if health == "misleading":
        if probe_used >= probe_budget:
            return VariantDecisionOutput(
                action="abandon",
                reason="misleading_quality_budget_exhausted",
            )
        return VariantDecisionOutput(
            action="rescue",
            reason="misleading_quality",
            next_variant_hint={"variant_kind": "precision", "action": "narrow"},
        )

    if probe_used >= probe_budget and not has_signal:
        all_exhausted = _all_planned_exhausted(inp.experiment_state)
        if all_exhausted:
            return VariantDecisionOutput(
                action="abandon",
                reason="all_variants_exhausted_without_signal",
            )
        return VariantDecisionOutput(
            action="abandon",
            reason="probe_budget_exhausted_no_signal",
        )

    if _has_split_evidence(variant):
        return VariantDecisionOutput(
            action="split",
            reason="distinct_sub_populations_detected",
            next_variant_hint={"variant_kind": "keyword_focus"},
        )

    return VariantDecisionOutput(
        action="continue",
        reason="within_budget_unclassified",
    )


def _all_planned_exhausted(state: "LinkedInExperimentState") -> bool:
    for vid in state.planned_variant_ids:
        v = state.variants.get(vid)
        if v and v.status in ("planned", "probing", "active"):
            return False
    return True


def _has_split_evidence(variant: "LinkedInSearchVariant") -> bool:
    """Detect evidence of distinct productive sub-populations.

    Requires page insights with at least two title clusters that each
    have signal, suggesting the variant covers separable populations.
    """
    if variant.settled_positive_count < 2:
        return False
    insights = variant.last_page_insights
    if not insights:
        return False
    clusters = insights.title_clusters
    if len(clusters) < 2:
        return False
    signal_clusters = sum(
        1 for c in clusters
        if isinstance(c, dict) and c.get("signal_count", 0) > 0
    )
    return signal_clusters >= 2
