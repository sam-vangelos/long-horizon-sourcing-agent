"""Tail ``run_log.jsonl`` streams and classify significance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from cloris.control_plane import state_dirs_for_brief_id

@dataclass(frozen=True)
class InternalLogEvent:
    source: str
    state_key: str
    event_type: str
    payload: dict[str, Any]
    raw_line: str


# LinkedIn-heavy today; widen as other modules adopt ``log_event`` consistently.
SIGNIFICANT_EVENT_TYPES = frozenset({
    "pipeline_start",
    "pipeline_end",
    "pipeline_error",
    "circuit_breaker",
    "bias_alert",
    "cadence_pause",
    "string_complete",
})


def is_significant_event_type(name: str | None) -> bool:
    n = (name or "").strip()
    return n in SIGNIFICANT_EVENT_TYPES


def parse_jsonl_records(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            yield rec


@dataclass
class RunLogTailCursor:
    path: Path
    offset_bytes: int = 0

    def drain_new_records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset_bytes:
            self.offset_bytes = 0
        if size == self.offset_bytes:
            return []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self.offset_bytes)
                chunk = fh.read()
                self.offset_bytes = fh.tell()
        except OSError:
            return []
        lines = chunk.splitlines()
        return list(parse_jsonl_records(lines))


def event_type(rec: dict[str, Any]) -> str | None:
    et = rec.get("event") or rec.get("type") or rec.get("event_type")
    if isinstance(et, str):
        return et
    return None


def summarize_significant_batches(
    events: list[InternalLogEvent],
) -> dict[str, Any]:
    """Bundle for narrator prompt."""

    return {
        "lines": [
            {
                "source": e.source,
                "state_key": e.state_key,
                "event": e.event_type,
                "payload_keys": sorted(
                    str(k)
                    for k in (e.payload or {}).keys()
                    if isinstance(k, str)
                )[:24],
            }
            for e in events[-20:]
        ],
    }


def build_tailers_for_brief(
    brief_id: str,
    *,
    state_root: Path | None = None,
) -> dict[str, RunLogTailCursor]:
    """Stable keys ``f\"{source}:{state_key}\"`` → tail cursor."""

    cursors: dict[str, RunLogTailCursor] = {}
    for source, state_dir in state_dirs_for_brief_id(
        brief_id.strip(), state_root=state_root
    ):
        key = f"{source}:{state_dir.name}"
        cursors[key] = RunLogTailCursor(path=state_dir / "run_log.jsonl")
    return cursors


def poll_significant_events(
    cursors: dict[str, RunLogTailCursor],
) -> list[InternalLogEvent]:
    """Read new JSONL rows and return those matching significance filter."""

    found: list[InternalLogEvent] = []
    for key, cur in cursors.items():
        parts = key.split(":", 1)
        source = parts[0]
        state_key = parts[1] if len(parts) > 1 else ""
        for rec in cur.drain_new_records():
            et = event_type(rec)
            if not is_significant_event_type(et):
                continue
            found.append(
                InternalLogEvent(
                    source=source,
                    state_key=state_key,
                    event_type=et or "",
                    payload=rec,
                    raw_line="",
                ),
            )
    return found


__all__ = [
    "InternalLogEvent",
    "RunLogTailCursor",
    "SIGNIFICANT_EVENT_TYPES",
    "build_tailers_for_brief",
    "event_type",
    "is_significant_event_type",
    "poll_significant_events",
    "summarize_significant_batches",
]
