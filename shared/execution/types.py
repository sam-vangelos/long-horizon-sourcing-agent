"""Shared execution types for candidate-stage orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateExecutionEnvelope:
    """Normalized per-candidate execution context shared across adapters."""

    source: str
    brief_id: str
    run_id: int
    work_unit_kind: str
    work_unit_source_id: str
    identity_key: str
    display_name: str
    profile_url: str
    snippet: Any | None = None
    source_cursor: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquisitionResult:
    """Normalized acquisition/evidence payload emitted by source adapters."""

    candidate: Any | None = None
    snippet: Any | None = None
    profile_summary: Any | None = None
    candidate_record: dict[str, Any] | None = None
    terminal_decision: str | None = None
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkUnitCheckpoint:
    """Small, explicit checkpoint payload emitted by source work-unit services."""

    status: str
    cursor: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SideEffectOutcome:
    """Durable record of a post-decision side effect."""

    effect_type: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)


# Backwards-compatible export name used by earlier phases.
SideEffectResult = SideEffectOutcome
