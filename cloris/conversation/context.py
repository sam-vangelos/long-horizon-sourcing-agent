"""Assemble recruiter-grounded telemetry context for conversation turns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cloris.control_plane import state_dirs_for_brief_id
from shared import output_paths
from shared.runtime_state import read_models


@dataclass(frozen=True)
class CitationRef:
    source: str
    state_key: str
    signal_ref: str


SAVE_CLASS = frozenset({"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE"})


def _tail_run_log_lines(log_path: Path, *, max_lines: int = 48) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    last = raw[-max_lines:] if len(raw) > max_lines else raw
    out: list[dict[str, Any]] = []
    for line in last:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
        except json.JSONDecodeError:
            continue
    return out


def _save_count(db_path: Path, run_id: int) -> int:
    decisions = read_models.run_decisions(db_path, run_id=run_id, limit=5000)
    n = 0
    for d in decisions:
        td = (d.terminal_decision or "").upper()
        if td in SAVE_CLASS or "SAVE" in td:
            n += 1
    return n


def build_live_context_and_citations(
    brief_id: str,
    *,
    state_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return (structured_context_jsonable, citations for debug payloads)."""

    pairs = state_dirs_for_brief_id(brief_id.strip(), state_root=state_root)
    orch = output_paths.resolve_orchestration_db_path()
    cos_row = read_models.chief_of_staff_run_by_brief(orch, brief_id=brief_id.strip())

    per_source: list[dict[str, Any]] = []
    citations: list[dict[str, str]] = []

    for source, state_dir in pairs:
        state_key = state_dir.name
        db_path = state_dir / "runtime_state.sqlite3"
        rid = read_models.latest_run_in_state_dir(db_path)
        run_detail = (
            read_models.run_by_id(db_path, run_id=rid) if rid is not None else None
        )
        log_path = state_dir / "run_log.jsonl"
        log_tail = _tail_run_log_lines(log_path)
        save_count = _save_count(db_path, rid) if rid is not None else 0

        block: dict[str, Any] = {
            "source": source,
            "state_key": state_key,
            "latest_run_id": rid,
            "run_status": run_detail.status if run_detail else None,
            "run_stop_reason": run_detail.stop_reason if run_detail else None,
            "run_started_at": run_detail.started_at if run_detail else None,
            "run_ended_at": run_detail.ended_at if run_detail else None,
            "save_class_count_latest_run": save_count,
            "recent_log_events": [
                {
                    "event": e.get("event") or e.get("type"),
                    "ts": e.get("ts") or e.get("timestamp"),
                }
                for e in log_tail[-12:]
            ],
        }
        per_source.append(block)
        citations.append(
            {
                "source": source,
                "state_key": state_key,
                "signal_ref": f"runtime_state.sqlite3:run_id={rid}",
            }
        )

    cos_block: dict[str, Any] | None = None
    if cos_row is not None:
        try:
            plan = json.loads(cos_row.dispatch_plan_json or "{}")
        except json.JSONDecodeError:
            plan = {}
        try:
            order = json.loads(cos_row.invocation_order_json or "[]")
        except json.JSONDecodeError:
            order = []
        cos_block = {
            "status": cos_row.status,
            "dispatch_plan": plan if isinstance(plan, dict) else {},
            "invocation_order": order if isinstance(order, list) else [],
            "started_at": cos_row.started_at,
        }
        citations.append(
            {
                "source": "orchestration",
                "state_key": brief_id,
                "signal_ref": "chief_of_staff_runs:latest",
            }
        )

    ctx: dict[str, Any] = {
        "brief_id": brief_id.strip(),
        "specialists": per_source,
        "chief_of_staff": cos_block,
    }
    return ctx, citations


def deterministic_status_answer(ctx: dict[str, Any]) -> str:
    """Operational fallback when LLM is unavailable — still grounded."""

    specs = ctx.get("specialists") or []
    if not specs:
        return (
            "No active state directories match this brief yet — nothing is running "
            "under this brief id in the workspace."
        )
    parts: list[str] = []
    for s in specs:
        name = str(s.get("source") or "source").replace("_", " ").title()
        st = s.get("run_status") or "unknown"
        saves = s.get("save_class_count_latest_run", 0)
        rid = s.get("latest_run_id")
        parts.append(
            f"{name}: latest run {rid}, status {st}, {saves} save-class "
            f"candidates on that run."
        )
    return " ".join(parts)
