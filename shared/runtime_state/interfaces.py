"""Typed interfaces for runtime-state bridges."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RuntimeStateBridge(Protocol):
    """Common bridge surface shared by source adapters."""

    output_dir: Path
    brief_id: str
    brief_name: str

    def has_runtime_state(self) -> bool: ...

    def start_or_resume_run(
        self,
        *,
        resume: bool,
        initial_progress: Any | None = None,
    ) -> tuple[int, Any]: ...

    def sync_progress(self, run_id: int, progress: Any) -> None: ...

    def load_progress(self, run_id: int) -> Any: ...

    def rebuild_artifacts(self, run_id: int) -> None: ...

    def record_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
    ) -> None: ...
