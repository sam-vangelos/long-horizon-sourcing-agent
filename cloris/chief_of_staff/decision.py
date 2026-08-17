"""Dispatch plan types for chief-of-staff orchestration.

Slice 2.5 introduces a deterministic dispatch planner whose output is
persisted in the orchestration SQLite as ``dispatch_plan_json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DispatchStep:
    """One module dispatch step in execution order."""

    module_name: str
    # Placeholder for v2 handoff gating (e.g., "only if confidence < 0.6").
    handoff_condition: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "module_name": self.module_name,
            "handoff_condition": self.handoff_condition,
        }


@dataclass(frozen=True)
class DispatchPlan:
    """Ordered dispatch plan for a brief's target modules."""

    steps: list[DispatchStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[dict[str, str | None]]]:
        return {"steps": [step.to_dict() for step in self.steps]}
