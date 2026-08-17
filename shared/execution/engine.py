"""Shared candidate execution engine."""

from __future__ import annotations

from shared.runtime_state.store import RuntimeStateStore

from .runtime import SharedExecutionRuntime
from .types import CandidateExecutionEnvelope


class CandidateExecutionEngine:
    """Thin facade over canonical candidate execution semantics."""

    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str,
        brief_id: str,
        source: str,
    ):
        self.runtime = SharedExecutionRuntime(
            store=store,
            output_dir=output_dir,
            brief_id=brief_id,
            source=source,
        )

    def envelope(self, **kwargs) -> CandidateExecutionEnvelope:
        return CandidateExecutionEnvelope(**kwargs)
