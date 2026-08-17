"""Fail-soft timing events for LinkedIn browser and checkpoint work."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from shared.storage import log_event


TimingRecorder = Callable[[str, dict[str, object]], None]

_NUMBER = (int, float)
TIMING_EVENT_SCHEMAS: Mapping[str, Mapping[str, type | tuple[type, ...]]] = {
    "card_focus_timing": {
        "elapsed_ms": _NUMBER,
        "card_index": int,
        "succeeded": bool,
    },
    "card_snapshot_timing": {
        "elapsed_ms": _NUMBER,
        "card_index": int,
        "text_chars": int,
        "succeeded": bool,
    },
    "profile_open_timing": {
        "elapsed_ms": _NUMBER,
        "succeeded": bool,
    },
    "profile_read_timing": {
        "elapsed_ms": _NUMBER,
        "pattern": str,
        "chunk_count": int,
        "wheel_events": int,
    },
    "profile_innertext_timing": {
        "elapsed_ms": _NUMBER,
        "text_chars": int,
        "succeeded": bool,
    },
    "profile_expand_timing": {
        "elapsed_ms": _NUMBER,
        "elements_walked": int,
        "clicks_made": int,
    },
    "checkpoint_progress_timing": {
        "elapsed_ms": _NUMBER,
        "lane_cost_reparse_ms": _NUMBER,
        "work_unit_rewrite_ms": _NUMBER,
        "projection_rebuild_ms": _NUMBER,
        "search_memory_reload_ms": _NUMBER,
    },
    "cadence_pause_timing": {
        "elapsed_ms": _NUMBER,
        "pause_seconds": _NUMBER,
    },
}


@dataclass(frozen=True)
class RunLogTimingRecorder:
    """Write only registered timing events to one run log."""

    path: Path

    def __call__(self, event: str, payload: dict[str, object]) -> None:
        if event == "card_focus_timing":
            log_event(self.path, "card_focus_timing", **payload)
        elif event == "card_snapshot_timing":
            log_event(self.path, "card_snapshot_timing", **payload)
        elif event == "profile_open_timing":
            log_event(self.path, "profile_open_timing", **payload)
        elif event == "profile_read_timing":
            log_event(self.path, "profile_read_timing", **payload)
        elif event == "profile_innertext_timing":
            log_event(self.path, "profile_innertext_timing", **payload)
        elif event == "profile_expand_timing":
            log_event(self.path, "profile_expand_timing", **payload)
        elif event == "checkpoint_progress_timing":
            log_event(self.path, "checkpoint_progress_timing", **payload)
        elif event == "cadence_pause_timing":
            log_event(self.path, "cadence_pause_timing", **payload)
        else:
            raise KeyError(event)


def emit_timing_event(
    recorder: TimingRecorder | None,
    event: str,
    **payload: object,
) -> None:
    """Validate and emit without ever affecting the primary path."""

    if recorder is None:
        return
    try:
        schema = TIMING_EVENT_SCHEMAS[event]
        for field, expected_type in schema.items():
            if field not in payload or not isinstance(
                payload[field], expected_type
            ):
                raise TypeError(f"invalid {event}.{field}")
        recorder(event, dict(payload))
    except BaseException:
        return
