"""Shared candidate execution primitives."""

from .engine import CandidateExecutionEngine
from .runtime import SharedExecutionRuntime
from .types import (
    AcquisitionResult,
    CandidateExecutionEnvelope,
    SideEffectOutcome,
    SideEffectResult,
    WorkUnitCheckpoint,
)

__all__ = [
    "AcquisitionResult",
    "CandidateExecutionEngine",
    "CandidateExecutionEnvelope",
    "SharedExecutionRuntime",
    "SideEffectOutcome",
    "SideEffectResult",
    "WorkUnitCheckpoint",
]
