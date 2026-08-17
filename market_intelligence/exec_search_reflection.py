"""Executive Search reflection / hunk composer (audit Move #9).

Slice-1 scope: defines the hunk shape now even though the wider
Executive Search module is mostly a stub. Audit plan rationale:
"the hunk schema is defensible" before the module's signal surface
hardens. Once the off-LinkedIn adapters (Crunchbase / News /
PitchBook) start populating real signal density, the composer's
threshold logic can tighten without touching the schema.

Hunk taxonomy:

- ``widen_company_stage_signals`` (section: ``company_stage_signals``)
  — fires when the brief carries a company_stage_signals list and the
  run produced fewer saves than the recruiter would expect at that
  stage. The proposed ``after`` is the union of the current list +
  one canonical adjacent stage.
- ``broaden_board_signals`` (section: ``board_signals``) — fires when
  the brief carries board_signals and the run shows facial-no
  clusters citing board-related signals. Proposed ``after`` is the
  current list + a "see board observer roles" guidance string the
  recruiter can shape into a concrete signal.

Failure-mode posture: empty inputs ⇒ empty hunk list. The composer
NEVER raises — Executive Search is a Slice-1 surface and reflection
mustn't crash the wider Gate-2 pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from market_intelligence.schema import MarketEvidenceBatch
from shared.storage import read_json, read_jsonl, write_json

logger = logging.getLogger(__name__)


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
FACIAL_NO_DECISIONS = {"FACIAL_NO"}


def maybe_build_and_persist_exec_search_research_packet(
    batch: MarketEvidenceBatch,
) -> MarketEvidenceBatch:
    """Attach exec-search dossier/adaptation evidence for market intelligence."""

    if batch.source != "exec_search":
        return batch

    output_dir = Path(batch.output_dir)
    run_log = read_jsonl(output_dir / "run_log.jsonl")
    investigation_path = output_dir / "investigation_packet.json"
    investigation_packet = {}
    if investigation_path.exists():
        try:
            investigation_packet = read_json(investigation_path)
        except (OSError, ValueError):
            investigation_packet = {}
    adaptation_events = [
        event for event in run_log if event.get("event") == "adaptation_decision"
    ]
    budget_events = [
        event for event in run_log if event.get("event") == "budget_exhausted"
    ]
    packet = {
        "context_metadata": {
            "source": "exec_search",
            "run_ref": batch.run_ref,
            "brief_version": batch.brief_version,
            "generated_at": batch.generated_at,
            "context_quality": "runtime_evidence",
            "analysis_provenance": (
                "exec_search_reflection."
                "maybe_build_and_persist_exec_search_research_packet"
            ),
        },
        "investigation_packet": investigation_packet,
        "dossier_signal_summary": {
            "runtime_summary": batch.runtime_summary,
            "metrics_summary": batch.metrics_summary,
            "budget_events": budget_events,
        },
        "adaptation_timeline": adaptation_events,
    }
    path = output_dir / "exec-search-research-input.json"
    try:
        write_json(path, packet)
    except OSError:
        pass
    else:
        batch.research_input_path = str(path)
    batch.research_context = packet
    batch.context_quality = "runtime_evidence"
    batch.analysis_provenance = packet["context_metadata"]["analysis_provenance"]
    return batch

# Audit Move #25: mirror briefing_polish._emit_stage so per-module
# reflection composers' stage logs interleave cleanly with the wider
# market-intel run trace.
def _emit_stage(message: str) -> None:
    """Mirror briefing_polish._emit_stage."""

    import sys

    print(f"[market-intel] {message}", file=sys.stderr, flush=True)


# Adjacent stages to propose when the recruiter's company_stage_signals
# list looks too narrow. Order matters editorially — we propose the
# closest adjacent first.
_ADJACENT_STAGE_LADDER: tuple[str, ...] = (
    "growth_stage",
    "late_stage",
    "public_company",
    "newly_public",
    "post_ipo_stagnant",
    "private_equity_owned",
)


def propose_exec_search_hunks(
    *,
    final_judgments: list[dict] | None,
    brief_raw: dict | None = None,
) -> list[dict]:
    """Compose Gate-2 hunks from an exec_search run's final-judgments.

    Brief snapshot is required to surface "before" values for the
    structured fields (``company_stage_signals``, ``board_signals``).
    Empty inputs ⇒ empty hunk list.
    """

    judgments = final_judgments or []
    brief_raw = brief_raw or {}

    if not judgments:
        _emit_stage(
            "reflection.exec_search:start judgments=0 result=empty_input"
        )
        return []

    _emit_stage(
        f"reflection.exec_search:start judgments={len(judgments)}"
    )

    saves_total = sum(
        1
        for row in judgments
        if isinstance(row, dict) and _decision_of(row) in SAVE_DECISIONS
    )
    facial_no_total = sum(
        1
        for row in judgments
        if isinstance(row, dict) and _decision_of(row) in FACIAL_NO_DECISIONS
    )
    surfaced_total = saves_total + facial_no_total

    hunks: list[dict] = []

    # Hunk 1: widen_company_stage_signals when discovery is thin and
    # the brief carries a small company_stage_signals list.
    current_stages = [
        s
        for s in (brief_raw.get("company_stage_signals") or [])
        if isinstance(s, str)
    ]
    if (
        0 < len(current_stages) < 3
        and surfaced_total > 0
        and saves_total < max(3, surfaced_total // 4)
    ):
        next_stage = _next_adjacent_stage(current_stages)
        if next_stage:
            proposed = list(current_stages) + [next_stage]
            hunks.append(
                _hunk(
                    hunk_id="exec-search-widen-company-stage-signals",
                    section="company_stage_signals",
                    kind="modify",
                    label="Widen company stage signals",
                    before=", ".join(current_stages),
                    after=", ".join(proposed),
                    rationale=(
                        f"Of {surfaced_total} candidates surfaced, only "
                        f"{saves_total} cleared at the declared stage "
                        f"signals ({', '.join(current_stages)}). "
                        f"Adding {next_stage!r} broadens the off-LinkedIn "
                        "signal surface to one adjacent stage."
                    ),
                    confidence=0.55,
                    target_field="company_stage_signals",
                )
            )

    # Hunk 2: broaden_board_signals when board_signals is set and
    # facial-no's cluster around board-adjacent profiles. We can't
    # detect "board-adjacent" from final_judgments alone in Slice 1,
    # so this hunk fires conservatively: only when board_signals is
    # set AND saves_total is unusually low for a non-trivial run.
    current_board = [
        s for s in (brief_raw.get("board_signals") or []) if isinstance(s, str)
    ]
    if (
        current_board
        and surfaced_total >= 10
        and saves_total < max(2, surfaced_total // 5)
    ):
        hunks.append(
            _hunk(
                hunk_id="exec-search-broaden-board-signals",
                section="board_signals",
                kind="modify",
                label="Broaden board signals",
                before=", ".join(current_board),
                after=", ".join(current_board)
                + ", board observer roles, advisory positions",
                rationale=(
                    f"{saves_total} of {surfaced_total} candidates "
                    "cleared at the declared board signals. The off-"
                    "LinkedIn adapters surface advisory and observer "
                    "roles in addition to formal directorships; "
                    "consider weighting those in."
                ),
                confidence=0.5,
                target_field="board_signals",
            )
        )

    _emit_stage(
        f"reflection.exec_search:end surfaced={surfaced_total} "
        f"saves={saves_total} hunks_proposed={len(hunks)} "
        f"sections={sorted({h['section'] for h in hunks})}"
    )
    return hunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decision_of(row: dict) -> str | None:
    return (
        row.get("decision")
        or row.get("terminal_decision")
        or (row.get("full_decision") or {}).get("decision")
        or (row.get("facial_decision") or {}).get("decision")
    )


def _next_adjacent_stage(current: list[str]) -> str | None:
    """Pick the first ladder stage not already in ``current``."""

    current_norm = {s.strip().lower() for s in current if s.strip()}
    for candidate in _ADJACENT_STAGE_LADDER:
        if candidate.strip().lower() not in current_norm:
            return candidate
    return None


def _hunk(
    *,
    hunk_id: str,
    section: str,
    kind: str,
    label: str,
    before: str | None,
    after: str,
    rationale: str,
    confidence: float,
    target_field: str,
) -> dict:
    return {
        "hunk_id": hunk_id,
        "section": section,
        "kind": kind,
        "label": label,
        "before": before,
        "after": after,
        "rationale": rationale,
        "confidence": confidence,
        "default_approved": confidence >= 0.65,
        "target_field": target_field,
    }
