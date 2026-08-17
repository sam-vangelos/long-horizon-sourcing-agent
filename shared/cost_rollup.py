"""Cross-module per-run cost rollup — audit Move #10.

Aggregates module-emitted cost telemetry into a single sidecar
artifact so the run-summary surface can show "this run cost $X across
N modules" without re-invoking per-module APIs to recompute.

The contract:

- Each module orchestrator that incurs metered API spend emits a
  ``cost_usd`` field at run-end. The canonical surface is the
  ``pipeline_end`` event in the module's
  ``<state_dir>/run_log.jsonl`` (per audit Move #6's
  ``log_event`` discipline). Designer additionally produces a
  ``CostTelemetry`` rollup the run-end hook persists; this module
  reads either source.
- This module's :func:`aggregate_cost_for_run` walks a per-source
  state-dir map (``{"linkedin": Path, "github": Path, ...}``),
  reads each module's run-log + per-module rollup artifact (when
  present), and returns a :class:`CostRollup` summing across.
- The result is JSON-serialisable so the worker can persist it to
  ``<run_dir>/cost_rollup.json`` for downstream consumption by the
  run-summary surface.

Posture:

- Fail-soft: missing run_log, malformed JSON, empty state-dirs all
  yield zero contributions, never raise. The cost rollup is
  observability — its failure must not abort the run.
- Read-only: this module never mutates per-module state-dirs.
  Aggregation is a one-shot pass; the artifact lives in the run
  folder, not the per-source state-dir.
- No model-level normalization: the cost numbers are taken at face
  value from the emitting module. A module that emits stale
  estimates produces stale rollups; the fix lives in the emitter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


@dataclass
class ModuleCost:
    """One module's contribution to the rollup."""

    module: str
    cost_usd: float = 0.0
    sources: list[str] = field(default_factory=list)
    """The provenance trail — e.g., ``["pipeline_end", "designer_cost_telemetry"]``."""


@dataclass
class CostRollup:
    """Cross-module cost rollup for a single run."""

    total_usd: float = 0.0
    by_module: list[ModuleCost] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    """Modules whose state-dir was supplied but didn't surface a cost signal."""

    def to_dict(self) -> dict:
        return {
            "total_usd": round(self.total_usd, 5),
            "by_module": [
                {
                    "module": mc.module,
                    "cost_usd": round(mc.cost_usd, 5),
                    "sources": list(mc.sources),
                }
                for mc in self.by_module
            ],
            "missing": list(self.missing),
        }


def aggregate_cost_for_run(
    state_dirs: Mapping[str, Path],
) -> CostRollup:
    """Walk the per-module state-dirs and return the cross-module rollup.

    ``state_dirs`` maps a module name (``"linkedin"`` / ``"github"`` /
    ``"researcher"`` / ``"designer"`` / ``"exec_search"``) to its
    state-dir Path. Missing keys are dropped silently — only the
    modules the caller supplies contribute to the rollup.

    Returns a :class:`CostRollup` whose ``by_module`` list mirrors the
    insertion order of ``state_dirs`` so the run-summary surface can
    render a stable order across runs.
    """

    rollup = CostRollup()
    for module, state_dir in state_dirs.items():
        cost = _module_cost_from_state_dir(module, Path(state_dir))
        if cost is None:
            rollup.missing.append(module)
            continue
        rollup.by_module.append(cost)
        rollup.total_usd += cost.cost_usd
    rollup.total_usd = round(rollup.total_usd, 5)
    return rollup


def _sum_token_cost_log_usd(
    log_path: Path,
    *,
    run_id: int | str | None = None,
    provider_filter: str | None = None,
    exclude_rows_with: tuple[str, ...] = (),
    field_equals: dict[str, object] | None = None,
) -> float | None:
    """Sum ``estimated_cost_usd`` across a run's ``token-cost-log.jsonl``.

    P4.2: returns ``None`` — never an affirmative ``0.0`` — when the log is
    absent, empty, unreadable, or every row's cost estimate is unknown (rate
    lookup miss). Mirrors the "missing" semantics of
    :func:`_cost_from_pipeline_end`, which treats "no signal"
    as absence rather than zero spend.

    Shadow-judge honesty: ``exclude_rows_with=("shadow_stage",)`` lets the
    PRIMARY run cost exclude shadow-evaluation spend by the row discriminator
    that marks shadow calls. Provider identity is not enough after the GLM
    promotion: ``provider="fireworks"`` now also serves PRIMARY calls. A row
    without a provider field still counts as primary.

    ``field_equals`` (added for the full-eval shadow extension): every
    key/value pair must match the row exactly (``record.get(key) ==
    value``) for the row to count. Used to split shadow spend PER TIER —
    facial-shadow and full-eval-shadow calls both land as
    ``provider="fireworks"`` rows, distinguished only by the
    ``shadow_stage`` field (``"facial_shadow"`` / ``"full_shadow"``) shared/
    judger.py's ``_facial_shadow_call`` / ``_full_shadow_call`` tag onto
    ``usage_context`` — without this filter, each tier's ``shadow_cost_usd``
    would double-count the other tier's spend.
    """
    path = Path(log_path)
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    total = 0.0
    found_any = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        if run_id is not None and record.get("run_id") != run_id:
            continue
        row_provider = str(record.get("provider") or "")
        if provider_filter is not None and row_provider != provider_filter:
            continue
        if any(record.get(key) is not None for key in exclude_rows_with):
            continue
        if field_equals and any(
            record.get(key) != value for key, value in field_equals.items()
        ):
            continue
        cost = record.get("estimated_cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total += float(cost)
            found_any = True
    return round(total, 6) if found_any else None


def _cost_per_save_usd(cost_usd: float | None, saves: int) -> float | None:
    """Guard div-by-zero; omit (None) rather than fabricate a rate with no saves."""
    if cost_usd is None or not saves:
        return None
    return round(cost_usd / saves, 6)


def write_cost_rollup_sidecar(
    rollup: CostRollup,
    *,
    run_dir: Path,
    filename: str = "cost_rollup.json",
) -> Path:
    """Persist a :class:`CostRollup` next to the run's other artifacts.

    Returns the path written. Caller is responsible for ensuring
    ``run_dir`` exists; this helper does NOT mkdir defensively because
    cost-rollup writes happen at finalize time, by which point the
    run dir is already established.
    """

    path = Path(run_dir) / filename
    path.write_text(json.dumps(rollup.to_dict(), indent=2, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Per-module cost-source readers
# ---------------------------------------------------------------------------


def _module_cost_from_state_dir(
    module: str,
    state_dir: Path,
) -> ModuleCost | None:
    """Read whatever cost signal the module surfaces in its state-dir.

    Module-specific reader strategies (in order of preference):

    1. The module's run_log.jsonl carries a ``pipeline_end`` event
       with a ``cost_usd`` field. This is the audit-Move-6 canonical
       surface that researcher / github / designer all share.
    2. Designer additionally surfaces a CostTelemetry rollup at
       ``<state_dir>/cost_telemetry.json`` (Designer Slice 5+ writes
       this). When present and the run-log doesn't carry an
       aggregated cost, fall back to it.

    Returns ``None`` when neither surface yields a number — the
    caller treats that as "missing".
    """

    if not state_dir or not state_dir.exists():
        return None

    sources: list[str] = []
    cost_usd = 0.0
    found_signal = False

    pipeline_cost = _cost_from_pipeline_end(state_dir / "run_log.jsonl")
    if pipeline_cost is not None:
        cost_usd += pipeline_cost
        sources.append("run_log.pipeline_end.cost_usd")
        found_signal = True

    if module == "designer" and not found_signal:
        designer_cost = _designer_cost_telemetry(state_dir / "cost_telemetry.json")
        if designer_cost is not None:
            cost_usd += designer_cost
            sources.append("cost_telemetry.json.total_usd")
            found_signal = True

    if not found_signal:
        usage_cost = _sum_token_cost_log_usd(
            state_dir / "token-cost-log.jsonl",
            exclude_rows_with=("shadow_stage",),
        )
        if usage_cost is not None:
            cost_usd += usage_cost
            sources.append("token-cost-log.jsonl.estimated_cost_usd")
            found_signal = True

    if not found_signal:
        return None
    return ModuleCost(module=module, cost_usd=cost_usd, sources=sources)


def _cost_from_pipeline_end(run_log_path: Path) -> float | None:
    """Return the last ``pipeline_end.cost_usd`` from the module's run-log.

    Each ``pipeline_end`` re-reads the append-only ``token-cost-log.jsonl``
    and emits the **cumulative** whole-log total at finalize time. Resume /
    retry / day-cycle sessions therefore repeat an ever-growing total, not an
    incremental slice. Summing every ``pipeline_end.cost_usd`` double-counts
    spend across sessions; the final row is the run's true cumulative cost.

    Rows that omit ``cost_usd`` (log absent, empty, or all unknown rates) are
    skipped — never treated as ``0.0``. Returns ``None`` when no
    ``pipeline_end`` row carries a numeric ``cost_usd``.
    """

    if not run_log_path.exists():
        return None
    last_cost: float | None = None
    try:
        for line in run_log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "pipeline_end":
                continue
            cost = event.get("cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                last_cost = float(cost)
    except OSError:
        return None
    return last_cost


def _designer_cost_telemetry(path: Path) -> float | None:
    """Read Designer's ``cost_telemetry.json`` written by run-end hook.

    Schema::

        {
          "primary_pass_usd": 1.23,
          "cross_check_usd": 0.45,
          "total_usd": 1.68
        }

    We read ``total_usd`` directly when present; otherwise sum the
    two pass-specific fields. Returns None on any failure.
    """

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    total = payload.get("total_usd")
    if isinstance(total, (int, float)):
        return float(total)
    primary = payload.get("primary_pass_usd")
    cross = payload.get("cross_check_usd")
    if isinstance(primary, (int, float)) and isinstance(cross, (int, float)):
        return float(primary) + float(cross)
    return None


__all__ = [
    "CostRollup",
    "ModuleCost",
    "_cost_per_save_usd",
    "_sum_token_cost_log_usd",
    "aggregate_cost_for_run",
    "write_cost_rollup_sidecar",
]
