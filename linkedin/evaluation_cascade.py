"""Shadow profile-probe cascade for evaluation cost optimization (D6-D8).

Adds a middle evaluation gate between profile extraction and full Opus
eval. Runs in shadow mode only — records decisions without suppressing
anything. D7 adds audit metrics; D8 adds activation criteria.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ProbeDecision(enum.Enum):
    FULL_EVAL = "full_eval"
    REVIEW_INFERRED = "review_inferred"
    REVIEW_FLAGGED = "review_flagged"
    REJECT_WITHOUT_OPUS = "reject_without_opus"


@dataclass(frozen=True)
class CascadeDecision:
    probe_decision: ProbeDecision
    confidence: float = 0.0
    rationale: str = ""
    probe_model: str | None = None
    shadow: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_decision": self.probe_decision.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "probe_model": self.probe_model,
            "shadow": self.shadow,
        }


@dataclass
class CascadePolicy:
    shadow_enabled: bool = True
    active: bool = False

    def is_shadow_mode(self) -> bool:
        return self.shadow_enabled and not self.active


@dataclass
class CascadeRecord:
    candidate_name: str = ""
    profile_url: str = ""
    lane_id: str = ""
    probe_decision: str = ""
    full_eval_decision: str = ""
    probe_confidence: float = 0.0
    full_eval_confidence: float | None = 0.0
    agreement: bool = True
    probe_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "profile_url": self.profile_url,
            "lane_id": self.lane_id,
            "probe_decision": self.probe_decision,
            "full_eval_decision": self.full_eval_decision,
            "probe_confidence": self.probe_confidence,
            "full_eval_confidence": self.full_eval_confidence,
            "agreement": self.agreement,
            "probe_rationale": self.probe_rationale,
        }


def _profile_probe_text(profile_summary: dict[str, Any]) -> str:
    """Flatten the fields used for capability matching.

    Mirrors ``CandidateProfileSummary.to_dict()`` (shared/schemas.py) —
    headline/experiences/education/skills_snippet. The probe used to read
    ``current_title``/``summary``, keys that dataclass never emits, so any
    signal living outside the headline was invisible to the probe.
    """

    parts: list[str] = [str(profile_summary.get("headline", ""))]
    for exp in profile_summary.get("experiences") or []:
        if not isinstance(exp, dict):
            continue
        parts.append(str(exp.get("title", "")))
        parts.append(str(exp.get("company", "")))
        parts.extend(str(bullet) for bullet in exp.get("summary_bullets") or [])
    for edu in profile_summary.get("education") or []:
        if not isinstance(edu, dict):
            continue
        parts.append(str(edu.get("degree", "")))
        parts.append(str(edu.get("school", "")))
        parts.append(str(edu.get("field", "")))
    parts.extend(str(skill) for skill in profile_summary.get("skills_snippet") or [])
    return " ".join(parts).lower()


class ProfileProbe:
    """Lightweight probe that produces a cascade decision from a profile summary.

    In shadow mode the probe runs but full_judge still executes unconditionally.
    The probe uses a cheaper model and a lightweight rubric derived from the brief.
    """

    def __init__(self, policy: CascadePolicy | None = None):
        self.policy = policy or CascadePolicy()
        self.shadow_records: list[CascadeRecord] = []

    def evaluate(
        self,
        profile_summary: dict[str, Any],
        brief_signals: dict[str, Any] | None = None,
    ) -> CascadeDecision:
        """Run probe evaluation on assembled profile summary.

        This is a heuristic probe — it does not call an LLM. It checks
        whether the profile has signals that clearly match or clearly
        don't match the brief's capability areas and non-fit patterns.
        A future iteration can upgrade this to a cheap LLM call.
        """
        signals = brief_signals or {}
        capability_areas = signals.get("capability_areas", [])
        non_fit_patterns = signals.get("non_fit_patterns", [])

        combined = _profile_probe_text(profile_summary)
        # Capability matching intentionally runs on the joined blob — a
        # cross-field capability match can only escalate (FULL_EVAL /
        # REVIEW_*), never suppress, so the per-field scoping is deliberately
        # non-fit-only.
        label_fields = [
            str(profile_summary.get("headline", "")).lower(),
            *[
                str(exp.get("title", "")).lower()
                for exp in profile_summary.get("experiences") or []
                if isinstance(exp, dict)
            ],
        ]

        # Check for non-fit patterns
        for nfp in non_fit_patterns:
            pattern_label = str(nfp if isinstance(nfp, str) else nfp.get("label", "")).lower()
            if pattern_label and any(pattern_label in field for field in label_fields):
                return CascadeDecision(
                    probe_decision=ProbeDecision.REJECT_WITHOUT_OPUS,
                    confidence=0.7,
                    rationale=f"Non-fit pattern match: {pattern_label}",
                    probe_model=None,
                    shadow=self.policy.is_shadow_mode(),
                )

        # Check for capability signal presence
        signal_matches = 0
        for cap in capability_areas:
            cap_text = str(cap if isinstance(cap, str) else cap.get("area", "")).lower()
            if cap_text and cap_text in combined:
                signal_matches += 1

        if not capability_areas or signal_matches == 0:
            return CascadeDecision(
                probe_decision=ProbeDecision.REVIEW_FLAGGED,
                confidence=0.4,
                rationale="No capability signal match in profile",
                probe_model=None,
                shadow=self.policy.is_shadow_mode(),
            )

        match_ratio = signal_matches / len(capability_areas) if capability_areas else 0
        if match_ratio >= 0.5:
            return CascadeDecision(
                probe_decision=ProbeDecision.FULL_EVAL,
                confidence=0.8,
                rationale=f"Strong capability match ({signal_matches}/{len(capability_areas)})",
                probe_model=None,
                shadow=self.policy.is_shadow_mode(),
            )

        return CascadeDecision(
            probe_decision=ProbeDecision.REVIEW_INFERRED,
            confidence=0.5,
            rationale=f"Partial capability match ({signal_matches}/{len(capability_areas)})",
            probe_model=None,
            shadow=self.policy.is_shadow_mode(),
        )

    def record_shadow_outcome(
        self,
        candidate_name: str,
        profile_url: str,
        lane_id: str,
        probe: CascadeDecision,
        full_eval_decision: str,
        full_eval_confidence: float | None,
    ) -> CascadeRecord:
        """Record the probe decision alongside the full eval decision for D7 audit."""
        probe_would_escalate = probe.probe_decision in (
            ProbeDecision.FULL_EVAL,
            ProbeDecision.REVIEW_INFERRED,
            ProbeDecision.REVIEW_FLAGGED,
        )
        full_eval_is_save = full_eval_decision.upper() in {
            "SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE",
        }
        # Agreement: if probe would have suppressed (reject) but full eval saved, that's disagreement
        agreement = not (not probe_would_escalate and full_eval_is_save)

        record = CascadeRecord(
            candidate_name=candidate_name,
            profile_url=profile_url,
            lane_id=lane_id,
            probe_decision=probe.probe_decision.value,
            full_eval_decision=full_eval_decision,
            probe_confidence=probe.confidence,
            full_eval_confidence=full_eval_confidence,
            agreement=agreement,
            probe_rationale=probe.rationale,
        )
        self.shadow_records.append(record)
        return record


# ---------------------------------------------------------------------------
# D7: Audit Sampling and False-Negative Guard
# ---------------------------------------------------------------------------


class AuditSampler:
    """Measures what the cascade would have skipped in shadow mode.

    In shadow mode every candidate gets both probe and full eval,
    so this is a natural 100% audit. Metrics are computed per-lane.
    """

    def __init__(self, records: list[CascadeRecord] | None = None, seed: int = 42):
        self._records = list(records or [])
        self._seed = seed

    @property
    def records(self) -> list[CascadeRecord]:
        return list(self._records)

    def add_records(self, records: list[CascadeRecord]) -> None:
        self._records.extend(records)

    def metrics(self, lane_id: str | None = None) -> dict[str, Any]:
        """Compute audit metrics, optionally filtered by lane."""
        subset = self._records
        if lane_id:
            subset = [r for r in self._records if r.lane_id == lane_id]

        total = len(subset)
        if total == 0:
            return {
                "total_candidates": 0,
                "false_negative_count": 0,
                "false_negative_rate": 0.0,
                "golden_save_preservation_rate": 1.0,
                "full_eval_suppression_opportunity": 0,
                "suppression_opportunity_rate": 0.0,
            }

        SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}

        # False negatives: probe would reject, but full eval saved
        false_negatives = [
            r for r in subset
            if r.probe_decision == ProbeDecision.REJECT_WITHOUT_OPUS.value
            and r.full_eval_decision.upper() in SAVE_DECISIONS
        ]

        # Golden saves: full eval saved — did probe also escalate?
        golden_saves = [r for r in subset if r.full_eval_decision.upper() in SAVE_DECISIONS]
        golden_preserved = [
            r for r in golden_saves
            if r.probe_decision != ProbeDecision.REJECT_WITHOUT_OPUS.value
        ]

        # Suppression opportunity: probe would have rejected
        suppressible = [
            r for r in subset
            if r.probe_decision == ProbeDecision.REJECT_WITHOUT_OPUS.value
        ]

        golden_count = len(golden_saves)
        golden_pres_rate = 1.0 if golden_count == 0 else round(len(golden_preserved) / golden_count, 4)
        return {
            "total_candidates": total,
            "false_negative_count": len(false_negatives),
            "false_negative_rate": round(len(false_negatives) / max(total, 1), 4),
            "golden_save_preservation_rate": golden_pres_rate,
            "full_eval_suppression_opportunity": len(suppressible),
            "suppression_opportunity_rate": round(len(suppressible) / max(total, 1), 4),
        }

    def per_lane_metrics(self) -> dict[str, dict[str, Any]]:
        lanes = {r.lane_id for r in self._records if r.lane_id}
        return {lane: self.metrics(lane_id=lane) for lane in sorted(lanes)}


# ---------------------------------------------------------------------------
# D8: Controlled Cascade Activation Criteria
# ---------------------------------------------------------------------------


@dataclass
class CascadeActivationPolicy:
    min_shadow_candidates: int = 200
    max_false_negative_rate: float = 0.02
    min_golden_save_preservation: float = 0.98
    per_lane_override: dict[str, bool] = field(default_factory=dict)
    active: bool = False

    def can_activate(self, sampler: AuditSampler) -> tuple[bool, list[str]]:
        """Check whether cascade activation is safe.

        Returns (permitted, blocking_reasons). Even when permitted,
        the cascade is off-by-default — requires explicit opt-in.
        """
        if not self.active:
            return False, ["active=False (default off)"]

        blocking: list[str] = []

        m = sampler.metrics()

        if m["total_candidates"] < self.min_shadow_candidates:
            blocking.append(
                f"Insufficient shadow data: {m['total_candidates']} < {self.min_shadow_candidates}"
            )

        if m["false_negative_rate"] > self.max_false_negative_rate:
            blocking.append(
                f"False-negative rate too high: {m['false_negative_rate']:.4f} > {self.max_false_negative_rate}"
            )

        if m["golden_save_preservation_rate"] < self.min_golden_save_preservation:
            blocking.append(
                f"Golden-save preservation too low: {m['golden_save_preservation_rate']:.4f} < {self.min_golden_save_preservation}"
            )

        return len(blocking) == 0, blocking
