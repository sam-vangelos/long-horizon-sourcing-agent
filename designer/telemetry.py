"""Designer module — per-run telemetry primitives.

Designer Slice 11. Aggregates the per-stage rates the spec §11
customer-launch readiness requires: facial pass rate, full-eval pass
rate, vision-evaluation pass rate, cross-check disagreement rate,
recruiter feedback marker distribution, image-misrepresentative
flag rate, per-customer monthly inference spend.

The aggregator is pure — given per-candidate stage outcomes + the
recruiter annotation stores from :mod:`designer.recruiter_annotations`
+ the vision evaluation cost telemetry, it returns a structured
:class:`DesignerRunTelemetry`. The wire layer (Slice 11 follow-up
that lands in `cloris/api.py`) renders this on the workspace
telemetry dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class StageOutcomeCounts:
    """Per-stage candidate counts. Aggregated from the
    `candidates` table's lifecycle history."""

    candidates_seen: int = 0
    facial_yes: int = 0
    facial_no: int = 0
    facial_borderline: int = 0
    full_save: int = 0
    full_reject: int = 0
    full_inferential_save: int = 0


@dataclass(frozen=True)
class VisionEvaluationCounts:
    """Vision-evaluation pipeline pass / fallback / hard-reject rollup."""

    candidates_evaluated: int = 0
    pass_count: int = 0  # judgment.fallback_reason == ""
    fallback_count: int = 0  # judgment.fallback_reason != ""
    hard_reject_count: int = 0  # fallback_reason starts with "hard_reject:"
    schema_invalid_count: int = 0
    image_grounding_failure_count: int = 0
    cross_check_total: int = 0
    cross_check_disagreement: int = 0


@dataclass(frozen=True)
class CostTelemetry:
    """Per-run cost rollup across vision-evaluation calls.

    ``primary_pass_usd`` covers the Gemini 2.5 Pro primary pass;
    ``cross_check_usd`` covers the Sonnet 4.6 cross-check pass.
    """

    primary_pass_usd: float = 0.0
    cross_check_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return round(self.primary_pass_usd + self.cross_check_usd, 5)


@dataclass(frozen=True)
class FeedbackMarkerTelemetry:
    """Recruiter feedback marker rollup across all principles."""

    useful_guidance_count: int = 0
    wrong_shallow_count: int = 0
    off_rubric_count: int = 0

    @property
    def total_count(self) -> int:
        return (
            self.useful_guidance_count
            + self.wrong_shallow_count
            + self.off_rubric_count
        )

    @property
    def useful_guidance_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return round(self.useful_guidance_count / self.total_count, 4)


@dataclass(frozen=True)
class DesignerRunTelemetry:
    """Composite per-run telemetry payload.

    Calculated rates are properties (not stored) so consumers see
    canonical values regardless of which counts the orchestrator
    happens to populate.
    """

    brief_state_key: str
    run_id: int | None
    stage: StageOutcomeCounts
    vision: VisionEvaluationCounts
    cost: CostTelemetry
    feedback: FeedbackMarkerTelemetry
    image_misrepresentative_count: int = 0

    @property
    def facial_pass_rate(self) -> float:
        seen = self.stage.facial_yes + self.stage.facial_no + self.stage.facial_borderline
        if seen == 0:
            return 0.0
        return round(self.stage.facial_yes / seen, 4)

    @property
    def full_eval_save_rate(self) -> float:
        evaluated = (
            self.stage.full_save
            + self.stage.full_reject
            + self.stage.full_inferential_save
        )
        if evaluated == 0:
            return 0.0
        return round(
            (self.stage.full_save + self.stage.full_inferential_save) / evaluated, 4
        )

    @property
    def vision_pass_rate(self) -> float:
        if self.vision.candidates_evaluated == 0:
            return 0.0
        return round(self.vision.pass_count / self.vision.candidates_evaluated, 4)

    @property
    def vision_fallback_rate(self) -> float:
        if self.vision.candidates_evaluated == 0:
            return 0.0
        return round(self.vision.fallback_count / self.vision.candidates_evaluated, 4)

    @property
    def cross_check_disagreement_rate(self) -> float:
        if self.vision.cross_check_total == 0:
            return 0.0
        return round(
            self.vision.cross_check_disagreement / self.vision.cross_check_total, 4
        )

    @property
    def image_misrepresentative_rate(self) -> float:
        if self.stage.full_save == 0:
            return 0.0
        return round(
            self.image_misrepresentative_count / self.stage.full_save, 4
        )


def aggregate_run_telemetry(
    *,
    brief_state_key: str,
    run_id: int | None,
    candidate_terminal_decisions: Iterable[str],
    facial_decisions: Iterable[str],
    vision_judgment_outcomes: Iterable[dict],
    feedback_marker_distribution: dict[str, dict[str, int]],
    excluded_asset_count: int,
) -> DesignerRunTelemetry:
    """Roll up per-run telemetry from raw orchestrator outputs.

    Pure function. Inputs are typed lists / dicts the orchestrator
    builds at run-end; outputs the structured :class:`DesignerRunTelemetry`
    the workspace surface consumes.
    """

    facial_outcomes = list(facial_decisions)
    facial_yes = sum(1 for d in facial_outcomes if d == "FACIAL_YES")
    facial_no = sum(1 for d in facial_outcomes if d == "FACIAL_NO")
    facial_borderline = sum(1 for d in facial_outcomes if d == "FACIAL_BORDERLINE")

    full_outcomes = list(candidate_terminal_decisions)
    full_save = sum(1 for d in full_outcomes if d == "SAVE")
    full_reject = sum(1 for d in full_outcomes if d == "REJECT")
    full_inferential_save = sum(1 for d in full_outcomes if d == "INFERENTIAL_SAVE")

    stage = StageOutcomeCounts(
        candidates_seen=len(facial_outcomes),
        facial_yes=facial_yes,
        facial_no=facial_no,
        facial_borderline=facial_borderline,
        full_save=full_save,
        full_reject=full_reject,
        full_inferential_save=full_inferential_save,
    )

    vision_outcomes_list = list(vision_judgment_outcomes)
    pass_count = 0
    fallback_count = 0
    hard_reject_count = 0
    schema_invalid_count = 0
    image_grounding_failure_count = 0
    cross_check_total = 0
    cross_check_disagreement = 0
    primary_pass_usd = 0.0
    cross_check_usd = 0.0

    for outcome in vision_outcomes_list:
        if not isinstance(outcome, dict):
            continue
        fallback_reason = str(outcome.get("fallback_reason") or "")
        if fallback_reason:
            fallback_count += 1
            if fallback_reason.startswith("hard_reject:"):
                hard_reject_count += 1
            elif fallback_reason.startswith("schema_invalid"):
                schema_invalid_count += 1
            elif fallback_reason.startswith("image_grounding"):
                image_grounding_failure_count += 1
        else:
            pass_count += 1

        cost = outcome.get("cost_estimate_usd")
        if isinstance(cost, (int, float)):
            primary_pass_usd += float(cost)

        cross_check = outcome.get("cross_check")
        if isinstance(cross_check, dict) and cross_check:
            cross_check_total += 1
            if outcome.get("cross_check_disagreement"):
                cross_check_disagreement += 1
            cross_check_cost = cross_check.get("cost_estimate_usd")
            if isinstance(cross_check_cost, (int, float)):
                cross_check_usd += float(cross_check_cost)

    vision = VisionEvaluationCounts(
        candidates_evaluated=len(vision_outcomes_list),
        pass_count=pass_count,
        fallback_count=fallback_count,
        hard_reject_count=hard_reject_count,
        schema_invalid_count=schema_invalid_count,
        image_grounding_failure_count=image_grounding_failure_count,
        cross_check_total=cross_check_total,
        cross_check_disagreement=cross_check_disagreement,
    )

    cost_payload = CostTelemetry(
        primary_pass_usd=round(primary_pass_usd, 5),
        cross_check_usd=round(cross_check_usd, 5),
    )

    useful_guidance = 0
    wrong_shallow = 0
    off_rubric = 0
    for markers in feedback_marker_distribution.values():
        if not isinstance(markers, dict):
            continue
        useful_guidance += int(markers.get("useful_guidance", 0))
        wrong_shallow += int(markers.get("wrong_shallow", 0))
        off_rubric += int(markers.get("off_rubric", 0))

    feedback = FeedbackMarkerTelemetry(
        useful_guidance_count=useful_guidance,
        wrong_shallow_count=wrong_shallow,
        off_rubric_count=off_rubric,
    )

    return DesignerRunTelemetry(
        brief_state_key=brief_state_key,
        run_id=run_id,
        stage=stage,
        vision=vision,
        cost=cost_payload,
        feedback=feedback,
        image_misrepresentative_count=excluded_asset_count,
    )
