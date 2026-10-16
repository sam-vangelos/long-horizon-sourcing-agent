"""Shared runtime bindings; execution owners hold their own mutable work state.

The coordinator establishes these bindings. Day-cycle event absence remains
meaningful, so late-bound fields retain init=False. There is no progress,
experiment map, candidate ledger, or page-control state in this context.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.brief_loader import Brief
    from shared.runtime_state import RuntimeStateStore, LinkedInRuntimeStateBridge
    from shared.governor import SessionGovernor
    from shared.bias_controls import BiasMonitor
    from linkedin.browser import LinkedInBrowser
    from linkedin.timing_telemetry import RunLogTimingRecorder

@dataclass
class EvaluationContext:
    test_mode: bool = field(init=False, repr=False)
    _bias_monitor: BiasMonitor | None = field(init=False, repr=False)
    _brief_id: str = field(init=False, repr=False)
    _governor: SessionGovernor = field(init=False, repr=False)
    _operator_stop_event: asyncio.Event = field(init=False, repr=False)
    _pause_requested: asyncio.Event = field(init=False, repr=False)
    _resume_event: asyncio.Event = field(init=False, repr=False)
    _runtime_bridge: LinkedInRuntimeStateBridge = field(init=False, repr=False)
    _runtime_run_id: int | None = field(init=False, repr=False)
    _runtime_state: RuntimeStateStore = field(init=False, repr=False)
    _session_expired: asyncio.Event = field(init=False, repr=False)
    _timing_recorder: RunLogTimingRecorder | None = field(init=False, repr=False)
    brief_obj: Brief = field(init=False, repr=False)
    browser: LinkedInBrowser = field(init=False, repr=False)
    log_path: Path = field(init=False, repr=False)
    output_dir: Path = field(init=False, repr=False)
    stats: dict[str, Any] = field(init=False, repr=False)
