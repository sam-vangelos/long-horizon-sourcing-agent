"""Read-only recruiter-facing live signal for active Cloris runs.

Canonical lifecycle (``runs.status`` in ``runtime_state.sqlite3``) is the
source of truth for whether the run is over (audit finding F-3). The
projection artifacts (``live-console.log``, ``run_log.jsonl``,
``execution_plan.json``, ``worker.json``) are only consulted for detail
enrichment and to classify the in-progress phase while a run is actually
``running``.

When canonical state disagrees with projection text — for example, the
projection still says ``Strategizing...`` but SQLite says the latest run
completed — we trust SQLite, force the phase/lifecycle to terminal, and
log a warning so the drift is visible in ``cloris.log`` rather than
misleading the recruiter.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloris.worker import is_pid_alive
from shared.runtime_state.read_models import RunSummary, latest_run_summary

log = logging.getLogger(__name__)

_TAIL_BYTES = 80_000
_JSONL_TAIL_BYTES = 96_000
_MAX_TEXT = 260
_MAX_STRINGS = 4
_MAX_EVENTS = 6

_LOW_LEVEL_ERROR_EVENTS = {
    "card_extract_error",
    "profile_browser_disconnect",
    "panel_close_browser_disconnect",
    "pipeline_error",
    "profile_error",
    "final_error",
}

# Phases that imply Cloris is doing live work right now. Used to detect
# canonical/projection drift: if SQLite says the run is terminal but the
# projection-classified phase is in this set, we override and log.
_ACTIVE_PHASES: frozenset[str] = frozenset(
    {
        "starting",
        "preparing",
        "working",
        "strategizing",
        "strategy_ready",
        "searching",
        "adapting",
        "reviewing",
        "writing_report",
        "recovering",
    }
)

# ``runs.status`` values that mean the latest run is over. ``running`` is
# the only ongoing status in the canonical table.
#
from shared.run_status_constants import TERMINAL_RUN_STATUSES

# Legacy projection artifacts may still say "errored"; treat as terminal
# when classifying drift even though it is not canonical runs.status.
_TERMINAL_RUN_STATUSES = TERMINAL_RUN_STATUSES | frozenset({"errored"})


def build_run_signal(
    state_dir: Path,
    *,
    source: str,
    state_key: str,
) -> dict[str, Any]:
    """Return a compact live signal payload for one state directory.

    Reads canonical lifecycle from ``runtime_state.sqlite3`` first
    (audit finding F-3); only consults projection artifacts when the
    canonical run is actually ``running``. Stale projection text from a
    previous run cannot announce a live phase once the canonical run
    has gone terminal.
    """

    console_text = _tail_text(state_dir / "live-console.log", max_bytes=_TAIL_BYTES)
    plan = _read_json(state_dir / "execution_plan.json")
    events = _read_jsonl_tail(state_dir / "run_log.jsonl", max_bytes=_JSONL_TAIL_BYTES)
    sidecar_alive = _worker_sidecar_alive(state_dir / "worker.json")
    canonical = _canonical_run_summary(state_dir)

    projection_phase, projection_headline, projection_detail = _phase_from_artifacts(
        console_text=console_text,
        events=events,
        plan=plan,
        active=sidecar_alive,
    )

    phase, headline, detail, active = _reconcile_with_canonical(
        canonical=canonical,
        sidecar_alive=sidecar_alive,
        projection_phase=projection_phase,
        projection_headline=projection_headline,
        projection_detail=projection_detail,
        source=source,
        state_key=state_key,
    )

    lifecycle = _lifecycle_from_phase(phase, active=active)
    plan_summary = _summarize_plan(plan)

    return {
        "source": source,
        "state_key": state_key,
        "active": active,
        "phase": phase,
        "lifecycle": lifecycle,
        "headline": headline,
        "detail": detail,
        **plan_summary,
        "recent_events": _summarize_events(events),
        "updated_at": _latest_mtime_iso(
            [
                state_dir / "live-console.log",
                state_dir / "run_log.jsonl",
                state_dir / "execution_plan.json",
                state_dir / "worker.json",
            ]
        ),
    }


def _canonical_run_summary(state_dir: Path) -> RunSummary | None:
    """Read canonical latest-run summary or ``None`` if the SQLite store
    is unreadable. Wraps :func:`latest_run_summary` so any defensive
    fall-back stays in one place."""

    db_path = state_dir / "runtime_state.sqlite3"
    try:
        return latest_run_summary(db_path)
    except Exception as exc:  # noqa: BLE001 — never let this error reach the wire
        log.debug(
            "live_signal: canonical lookup failed at %s: %s", db_path, exc
        )
        return None


def _reconcile_with_canonical(
    *,
    canonical: RunSummary | None,
    sidecar_alive: bool,
    projection_phase: str,
    projection_headline: str,
    projection_detail: str | None,
    source: str,
    state_key: str,
) -> tuple[str, str, str | None, bool]:
    """Return the final ``(phase, headline, detail, active)`` after
    cross-checking projection-derived phase against canonical SQLite
    lifecycle.

    Rules:

    - No canonical state available (legacy state dir / unreadable DB)
      → fall back to projection-only behavior so we don't regress on
      pre-runtime-state state dirs.
    - Canonical status is ``running`` → trust the projection-derived
      phase. Sidecar liveness still gates the ``active`` flag for the
      "PID died but reconciler hasn't run yet" window.
    - Canonical status is terminal → override phase to ``completed``,
      regardless of what projection text says. Log a warning when the
      projection-classified phase implied live work.
    - Canonical status is unknown / missing → trust the projection.
    """

    if canonical is None or canonical.status is None:
        return projection_phase, projection_headline, projection_detail, sidecar_alive

    status = canonical.status
    if status == "running":
        # Canonical agrees the run is alive. ``active`` requires both
        # a running canonical run AND a live sidecar PID — so a worker
        # that crashed before the reconciler marked it abandoned still
        # collapses to ``active=False`` from the sidecar check.
        return (
            projection_phase,
            projection_headline,
            projection_detail,
            sidecar_alive,
        )

    if status in _TERMINAL_RUN_STATUSES:
        if projection_phase in _ACTIVE_PHASES:
            # Drift: projections claim Cloris is mid-run, canonical
            # says the run is over. Trust SQLite and surface the
            # disagreement to ``cloris.log`` so it's investigable.
            log.warning(
                "live_signal: projection/canonical drift "
                "source=%s state_key=%s projection_phase=%s "
                "canonical_status=%s — overriding to 'completed'",
                source,
                state_key,
                projection_phase,
                status,
            )
        if status == "completed":
            return (
                "completed",
                "Finished this pass",
                "The latest sourcing pass has ended; the report and run artifacts are available.",
                False,
            )
        # Other terminal statuses (including ``succeeded``, ``error``,
        # ``failed``, ``abandoned``, ``governor_limit_reached``,
        # ``interrupted``, legacy ``errored``). Distinct enums live elsewhere;
        # neutral copy — run report surfaces detail.
        return (
            "completed",
            "Run ended",
            "The latest sourcing pass ended. See the run report for detail.",
            False,
        )

    # Unknown status: treat as projection-only.
    return projection_phase, projection_headline, projection_detail, sidecar_alive


def _tail_text(path: Path, *, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    try:
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            raw = fh.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl_tail(path: Path, *, max_bytes: int) -> list[dict[str, Any]]:
    text = _tail_text(path, max_bytes=max_bytes)
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _worker_sidecar_alive(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    pid = payload.get("pid")
    if not isinstance(pid, int):
        return False
    return is_pid_alive(pid)


def _phase_from_artifacts(
    *,
    console_text: str,
    events: list[dict[str, Any]],
    plan: dict[str, Any] | None,
    active: bool,
) -> tuple[str, str, str | None]:
    text = console_text.lower()
    event_names = [str(e.get("event") or e.get("event_type") or "") for e in events[-20:]]
    last_event = event_names[-1] if event_names else ""

    if active and (
        "report_started" in event_names[-5:]
        or "run_report_generated" in event_names[-5:]
        or "[market-intel]" in text
    ):
        return (
            "writing_report",
            "Writing up the run",
            "Cloris has finished the sourcing pass and is turning the work into report and market notes.",
        )
    if active and (
        "block_adaptation" in event_names[-5:]
        or "linkedin_block_exploitation" in event_names[-5:]
        or "architecture_pivot" in event_names[-5:]
    ):
        return (
            "adapting",
            "Adapting the search plan",
            "Cloris is using the first search block to promote productive lanes and demote weak ones.",
        )
    if active and (
        "strategy_started" in event_names[-5:]
        or ("strategizing..." in text and "strategy complete" not in text)
    ):
        return (
            "strategizing",
            "Strategizing",
            "Cloris is asking Opus to synthesize the search architecture and compound strings before touching LinkedIn.",
        )
    if active and (
        "strategy_completed" in event_names[-5:]
        or (plan is not None and "--- execution" not in text)
    ):
        return (
            "strategy_ready",
            "Strategy ready",
            "Cloris has a search plan and is preparing the LinkedIn execution order.",
        )
    if active and (
        "--- execution" in text
        or last_event
        in {
            "execution_started",
            "string_started",
            "string_results",
            "string_scouted",
            "glance_assess",
            "linkedin_search_assess",
            "candidate_opened",
            "candidate_saved",
            "string_completed",
            "string_complete",
        }
    ):
        return (
            "searching",
            "Searching LinkedIn",
            "Cloris is running the planned strings, scouting result quality, and opening candidates that clear the bar.",
        )
    if active:
        if "connected to browser" in text:
            return (
                "starting",
                "Opening the sourcing session",
                "Cloris is attaching to the controlled Chrome profile and loading the brief context.",
            )
        return (
            "working",
            "Working",
            "Cloris is active; more detailed signal will appear as the worker writes its next checkpoint.",
        )
    if event_names and event_names[-1] in {"pipeline_end", "run_snapshot_finalized", "market_intel_updated"}:
        return (
            "completed",
            "Finished this pass",
            "The latest sourcing pass has ended; the report and run artifacts are available.",
        )
    return ("idle", "No live signal", "There is no active worker for this card right now.")


def _lifecycle_from_phase(phase: str, *, active: bool) -> str:
    if phase in {"starting", "preparing", "working"}:
        return "preparing" if active else "ready"
    if phase in {"strategizing", "strategy_ready"}:
        return "strategizing"
    if phase in {"searching", "adapting"}:
        return "searching"
    if phase == "reviewing":
        return "reviewing"
    if phase == "writing_report":
        return "writing_report"
    if phase in {"completed", "finished"}:
        return "finished"
    if phase == "recovering":
        return "recovering"
    return "ready"


def _summarize_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {
            "strategy_rationale": None,
            "strategy_architecture": None,
            "strategy_architecture_rationale": None,
            "generated_string_count": None,
            "coverage_gap_count": None,
            "strategy_strings": [],
        }

    generated = plan.get("generated_strings")
    strings = generated if isinstance(generated, list) else []
    gaps = plan.get("coverage_gaps")
    coverage_gaps = gaps if isinstance(gaps, list) else []

    previews: list[dict[str, Any]] = []
    for idx, item in enumerate(strings[:_MAX_STRINGS], start=1):
        if not isinstance(item, dict):
            continue
        domain_lane = _clean_optional(item.get("domain_lane"))
        family_key = _clean_optional(item.get("family_key"))
        label = _clean_optional(item.get("name")) or _human_label(domain_lane or family_key) or f"String {idx}"
        previews.append(
            {
                "id": idx,
                "label": _truncate(label, 96),
                "rationale": _truncate(_clean_optional(item.get("rationale")), _MAX_TEXT),
                "boolean": _truncate(_clean_optional(item.get("boolean")), _MAX_TEXT),
                "domain_lane": domain_lane,
                "novelty_bucket": _clean_optional(item.get("novelty_bucket")),
            }
        )

    return {
        "strategy_rationale": _truncate(
            _clean_optional(plan.get("strategy_rationale")), 520
        ),
        "strategy_architecture": _human_label(_clean_optional(plan.get("architecture"))),
        "strategy_architecture_rationale": _truncate(
            _clean_optional(plan.get("architecture_rationale")), 360
        ),
        "generated_string_count": len(strings),
        "coverage_gap_count": len(coverage_gaps),
        "strategy_strings": previews,
    }


def _summarize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in reversed(events):
        event = _clean_optional(rec.get("event") or rec.get("event_type"))
        if not event:
            continue
        if event in _LOW_LEVEL_ERROR_EVENTS:
            continue
        summarized = _event_summary(event, rec)
        if summarized is None:
            continue
        out.append(summarized)
        if len(out) >= _MAX_EVENTS:
            break
    return out


def _event_summary(event: str, rec: dict[str, Any]) -> dict[str, Any] | None:
    ts = _clean_optional(rec.get("timestamp") or rec.get("created_at"))
    if event == "pipeline_start":
        return {
            "kind": "start",
            "label": "Started the sourcing pass",
            "detail": "Loaded the brief and opened the LinkedIn run.",
            "timestamp": ts,
        }
    if event == "strategy_started":
        return {
            "kind": "strategy",
            "label": "Started the search strategy",
            "detail": "Synthesizing the search architecture before opening LinkedIn.",
            "timestamp": ts,
        }
    if event == "strategy_completed":
        count = rec.get("generated_string_count")
        return {
            "kind": "strategy",
            "label": "Finished the search strategy",
            "detail": f"{count} planned strings are ready." if count is not None else None,
            "timestamp": ts,
        }
    if event == "execution_started":
        count = rec.get("string_count")
        return {
            "kind": "search",
            "label": "Started LinkedIn execution",
            "detail": f"{count} strings queued." if count is not None else None,
            "timestamp": ts,
        }
    if event == "string_started":
        sid = rec.get("string_id")
        return {
            "kind": "search",
            "label": f"Opened string #{sid}" if sid is not None else "Opened a search string",
            "detail": _truncate(_clean_optional(rec.get("rationale")), 180),
            "timestamp": ts,
        }
    if event == "string_results":
        sid = rec.get("string_id")
        count = _clean_optional(rec.get("result_count_text")) or _clean_optional(rec.get("result_count"))
        return {
            "kind": "search",
            "label": f"Opened string #{sid}" if sid is not None else "Opened a search string",
            "detail": f"LinkedIn returned {count} results." if count else None,
            "timestamp": ts,
        }
    if event == "glance_assess":
        sid = rec.get("string_id")
        action = _human_label(_clean_optional(rec.get("action")))
        confidence = rec.get("confidence")
        detail = action
        if isinstance(confidence, (int, float)):
            detail = f"{action or 'Scouted'} with {int(confidence * 100)}% confidence."
        return {
            "kind": "scout",
            "label": f"Scouted string #{sid}" if sid is not None else "Scouted a result page",
            "detail": detail,
            "timestamp": ts,
        }
    if event == "linkedin_search_assess":
        sid = rec.get("string_id")
        decision = _human_label(_clean_optional(rec.get("decision")))
        rationale = _truncate(_clean_optional(rec.get("rationale")), 180)
        return {
            "kind": "decision",
            "label": f"Chose next move for string #{sid}" if sid is not None else "Chose the next search move",
            "detail": " - ".join(part for part in [decision, rationale] if part),
            "timestamp": ts,
        }
    if event == "candidate_saved":
        name = _clean_optional(rec.get("name"))
        # P1.2: the honest emitter carries linkedin_save. A physically
        # failed save must not render as "Saved a candidate" in the live
        # feed. Absent flag (legacy logs) keeps the old label.
        if rec.get("linkedin_save") is False:
            return {
                "kind": "save",
                "label": "Save failed — will retry",
                "detail": name,
                "timestamp": ts,
            }
        return {
            "kind": "save",
            "label": "Saved a candidate",
            "detail": name,
            "timestamp": ts,
        }
    if event == "candidate_opened":
        name = _clean_optional(rec.get("name"))
        return {
            "kind": "review",
            "label": "Opened a candidate profile",
            "detail": name,
            "timestamp": ts,
        }
    if event == "report_started":
        return {
            "kind": "report",
            "label": "Started the run report",
            "detail": "Compiling the sourcing pass into a readable report.",
            "timestamp": ts,
        }
    if event == "report_completed":
        return {
            "kind": "report",
            "label": "Finished the run report",
            "detail": "The run report is ready to read.",
            "timestamp": ts,
        }
    if event in {"string_complete", "string_completed"}:
        sid = rec.get("string_id")
        saved = rec.get("saved")
        facial_yes = rec.get("facial_yes")
        return {
            "kind": "complete",
            "label": f"Finished string #{sid}" if sid is not None else "Finished a search string",
            "detail": f"{saved or 0} saved; {facial_yes or 0} candidates cleared initial review.",
            "timestamp": ts,
        }
    if event in {"linkedin_block_exploitation", "block_adaptation", "architecture_pivot"}:
        return {
            "kind": "adapt",
            "label": "Adapted the search plan",
            "detail": "Promoted productive lanes and re-ordered the remaining strings.",
            "timestamp": ts,
        }
    if event == "run_report_generated":
        return {
            "kind": "report",
            "label": "Wrote the run report",
            "detail": "Compiled the sourcing pass into a readable report.",
            "timestamp": ts,
        }
    if event == "market_intel_updated":
        return {
            "kind": "market",
            "label": "Updated the market read",
            "detail": "Folded this run into market intelligence.",
            "timestamp": ts,
        }
    if event == "pipeline_end":
        return {
            "kind": "finish",
            "label": "Finished the sourcing pass",
            "detail": _stats_detail(rec),
            "timestamp": ts,
        }
    return None


def _stats_detail(rec: dict[str, Any]) -> str | None:
    saved = rec.get("saved")
    rejected = rec.get("rejected")
    facial_yes = rec.get("facial_yes")
    parts: list[str] = []
    if isinstance(saved, int):
        parts.append(f"{saved} saved")
    if isinstance(rejected, int):
        parts.append(f"{rejected} rejected")
    if isinstance(facial_yes, int):
        parts.append(f"{facial_yes} candidates cleared initial review")
    return "; ".join(parts) if parts else None


def _latest_mtime_iso(paths: list[Path]) -> str | None:
    mtimes: list[float] = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), timezone.utc).isoformat()


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _human_label(value: str | None) -> str | None:
    if not value:
        return None
    return value.replace("_", " ").replace("-", " ").strip().title()


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "..."
