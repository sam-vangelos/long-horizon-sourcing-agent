"""Researcher-specific reflection / hunk composer (audit Move #9).

Mirrors :mod:`market_intelligence.github_reflection` in shape but
proposes hunks for the recruiter-authored fields the Researcher module
keeps under ``source_config.researcher`` plus the discipline-derived
floors (per :mod:`researcher.discipline_defaults`).

Two surfaces:

1. **Ecosystem narrative** (research-context packet) — same role as
   the github variant: a fail-soft observation summary writeable to
   ``researcher-research-input.json`` for downstream readers.
2. **Hunk composer** (``propose_researcher_hunks``) — pure function
   that consumes a final-judgments stream + the brief's current
   ``source_config.researcher`` settings and returns Gate-2 hunks
   (same dict shape as ``market_intelligence.reflection`` propose-phase
   hunks: ``hunk_id`` / ``section`` / ``kind`` / ``label`` / ``before``
   / ``after`` / ``rationale`` / ``confidence`` / ``default_approved``
   / ``target_field``).

Hunk taxonomy (per the audit plan's "lower_h_index_floor /
expand_conference_allowlist" exemplars):

- ``lower_h_index_floor`` (section: ``h_index_floor``) — fires when
  the run's facial-no rate clusters around the floor and the recruiter
  brief carries an explicit floor. The proposed ``after`` is two-thirds
  of the current floor (recruiter can edit before approving).
- ``expand_conference_allowlist`` (section: ``conference_allowlist``)
  — fires when discovered candidates published at venues outside the
  allowlist at a rate >25% of facial-yes counts. The proposed ``after``
  is the union of the current allowlist + the top venues observed.
- ``broaden_research_topics`` (section: ``research_topics``) — fires
  when the run discovered <10 candidates and the brief has fewer than
  3 research topics (signal: query surface area too narrow).

Failure-mode posture: every read is fail-soft. Empty inputs ⇒ empty
hunk list. The wider reflection flow doesn't depend on hunks being
non-empty; they're proposals, not control state.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from market_intelligence.schema import MarketEvidenceBatch
from shared.storage import read_jsonl, write_json

logger = logging.getLogger(__name__)


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
FACIAL_NO_DECISIONS = {"FACIAL_NO"}

MAX_TOP_VENUES = 10
DEFAULT_FLOOR_LOWER_FRACTION = 2 / 3


# Audit Move #25: mirror briefing_polish._emit_stage so per-module
# reflection composers' stage logs interleave cleanly with the wider
# market-intel run trace. Stage names use a `reflection.researcher:`
# prefix so a downstream parser can route them to the right module.
def _emit_stage(message: str) -> None:
    """Mirror briefing_polish._emit_stage."""

    import sys

    print(f"[market-intel] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Research-context packet (mirror of github_reflection's variant)
# ---------------------------------------------------------------------------


def maybe_build_and_persist_researcher_research_packet(
    batch: MarketEvidenceBatch,
) -> MarketEvidenceBatch:
    """Build + persist a researcher-source research packet, in place on ``batch``.

    Returns the batch unchanged when ``batch.source != "researcher"``.
    """

    if batch.source != "researcher":
        return batch

    output_dir = Path(batch.output_dir)
    research_input_path = output_dir / "researcher-research-input.json"
    final_judgments = batch.final_judgments or _load_optional_jsonl(
        output_dir / "final_judgments.jsonl"
    )

    narrative = build_researcher_ecosystem_narrative(
        final_judgments=final_judgments,
    )

    packet = {
        "context_metadata": {
            "source": "researcher",
            "run_ref": batch.run_ref,
            "brief_version": batch.brief_version,
            "generated_at": batch.generated_at,
            "context_quality": _context_quality_label(narrative),
            "analysis_provenance": (
                "researcher_reflection."
                "maybe_build_and_persist_researcher_research_packet"
            ),
        },
        "ecosystem_narrative": narrative,
        "metrics_summary": batch.metrics_summary,
    }

    try:
        write_json(research_input_path, packet)
    except OSError as exc:
        logger.warning(
            "researcher_reflection: failed to persist packet at %s (%s); "
            "narrative attached to batch without disk write",
            research_input_path,
            exc,
        )
    else:
        batch.research_input_path = str(research_input_path)
    batch.research_context = packet
    batch.context_quality = packet["context_metadata"]["context_quality"]
    batch.analysis_provenance = packet["context_metadata"]["analysis_provenance"]
    return batch


def build_researcher_ecosystem_narrative(
    *,
    final_judgments: list[dict] | None,
) -> dict[str, Any]:
    """Pure transform: final_judgments → ecosystem narrative dict."""

    judgments = final_judgments or []
    save_records = list(_iter_records_with_decision(judgments, SAVE_DECISIONS))
    facial_no_records = list(
        _iter_records_with_decision(judgments, FACIAL_NO_DECISIONS)
    )

    by_venue: Counter = Counter()
    by_country: Counter = Counter()
    h_indices: list[int] = []
    papers_in_window: list[int] = []

    for record in save_records:
        candidate = _candidate_record(record)
        if not candidate:
            continue
        for venue in candidate.get("top_venues") or []:
            if isinstance(venue, str) and venue:
                by_venue[venue] += 1
        for affiliation in candidate.get("affiliations") or []:
            if isinstance(affiliation, str):
                country = _extract_country_code(affiliation)
                if country:
                    by_country[country] += 1
        h_index = candidate.get("h_index")
        if isinstance(h_index, int):
            h_indices.append(h_index)
        piw = candidate.get("papers_in_window")
        if isinstance(piw, int):
            papers_in_window.append(piw)

    # Facial-no near-floor: count candidates that fell just below the
    # h_index floor (within 1/3 of the floor). Empty when no h_index
    # data is available — the hunk composer treats it as "insufficient
    # signal to propose a lower floor."
    facial_no_h_indices: list[int] = []
    for record in facial_no_records:
        candidate = _candidate_record(record)
        if not candidate:
            continue
        h_index = candidate.get("h_index")
        if isinstance(h_index, int):
            facial_no_h_indices.append(h_index)

    return {
        "totals": {
            "saves_total": len(save_records),
            "facial_no_total": len(facial_no_records),
        },
        "top_venues": [
            {"venue": v, "count": c} for v, c in by_venue.most_common(MAX_TOP_VENUES)
        ],
        "by_country": dict(by_country),
        "h_index_distribution": {
            "saved": _quartile_summary(h_indices),
            "facial_no": _quartile_summary(facial_no_h_indices),
        },
        "papers_in_window_distribution": _quartile_summary(papers_in_window),
    }


# ---------------------------------------------------------------------------
# Hunk composer (audit Move #9)
# ---------------------------------------------------------------------------


def propose_researcher_hunks(
    *,
    final_judgments: list[dict] | None,
    brief_raw: dict | None = None,
) -> list[dict]:
    """Compose Gate-2 hunks from a researcher run's final-judgments.

    Brief snapshot is optional; when supplied, the composer can emit
    "before" values for hunks that propose modifying an existing floor.
    Empty inputs ⇒ empty hunk list.
    """

    judgments = final_judgments or []
    if not judgments:
        _emit_stage(
            "reflection.researcher:start judgments=0 result=empty_input"
        )
        return []

    _emit_stage(
        f"reflection.researcher:start judgments={len(judgments)}"
    )

    narrative = build_researcher_ecosystem_narrative(
        final_judgments=judgments
    )
    saves_total = narrative["totals"]["saves_total"]
    facial_no_total = narrative["totals"]["facial_no_total"]

    _emit_stage(
        f"reflection.researcher:narrative_built saves={saves_total} "
        f"facial_no={facial_no_total} top_venues={len(narrative.get('top_venues') or [])}"
    )

    source_config = (
        ((brief_raw or {}).get("source_config") or {}).get("researcher") or {}
    )

    hunks: list[dict] = []

    # Hunk 1: lower_h_index_floor when the facial-no rate is dense
    # near the floor and the brief carries an explicit floor.
    current_floor = source_config.get("h_index_floor")
    if (
        isinstance(current_floor, int)
        and current_floor > 3
        and facial_no_total >= max(saves_total, 5)
    ):
        proposed_floor = max(1, int(current_floor * DEFAULT_FLOOR_LOWER_FRACTION))
        if proposed_floor < current_floor:
            hunks.append(
                _hunk(
                    hunk_id="researcher-lower-h-index-floor",
                    section="h_index_floor",
                    kind="modify",
                    label="Lower h-index floor",
                    before=str(current_floor),
                    after=str(proposed_floor),
                    rationale=(
                        f"Of {facial_no_total + saves_total} candidates "
                        f"surfaced, {facial_no_total} were rejected at "
                        "facial. Lowering the floor lets the full evaluator "
                        "see candidates the recruiter may want without "
                        "a triage gate they can't observe."
                    ),
                    confidence=0.6,
                    target_field="source_config.researcher.h_index_floor",
                )
            )

    # Hunk 2: expand_conference_allowlist when saved candidates publish
    # at venues outside the current allowlist with notable frequency.
    current_allowlist_raw = source_config.get("conference_allowlist") or []
    current_allowlist = {
        v.strip().lower()
        for v in current_allowlist_raw
        if isinstance(v, str) and v.strip()
    }
    top_venues = narrative.get("top_venues") or []
    out_of_allowlist = [
        item
        for item in top_venues
        if item["venue"].strip().lower() not in current_allowlist
    ]
    if (
        current_allowlist
        and saves_total > 0
        and out_of_allowlist
        and out_of_allowlist[0]["count"] >= max(2, saves_total // 4)
    ):
        proposed_additions = [item["venue"] for item in out_of_allowlist[:5]]
        proposed = list(current_allowlist_raw) + proposed_additions
        hunks.append(
            _hunk(
                hunk_id="researcher-expand-conference-allowlist",
                section="conference_allowlist",
                kind="modify",
                label="Expand conference allowlist",
                before=", ".join(current_allowlist_raw)
                if current_allowlist_raw
                else None,
                after=", ".join(proposed),
                rationale=(
                    f"Saved candidates published at {len(out_of_allowlist)} "
                    "venues outside the current allowlist. The top "
                    f"{len(proposed_additions)} venues account for the "
                    "bulk of this volume — consider weighting them in."
                ),
                confidence=0.55,
                target_field="source_config.researcher.conference_allowlist",
            )
        )

    # Hunk 3: broaden_research_topics when discovery is thin and the
    # brief carries a small topic surface.
    current_topics_raw = source_config.get("research_topics") or []
    current_topics = [t for t in current_topics_raw if isinstance(t, str)]
    if (
        saves_total + facial_no_total < 10
        and 0 < len(current_topics) < 3
    ):
        hunks.append(
            _hunk(
                hunk_id="researcher-broaden-research-topics",
                section="research_topics",
                kind="modify",
                label="Broaden research topics",
                before=", ".join(current_topics),
                after=", ".join(current_topics) + ", <add 1-2 adjacent topics>",
                rationale=(
                    f"Only {saves_total + facial_no_total} candidates "
                    f"surfaced across {len(current_topics)} declared "
                    "topic(s). Adding 1-2 adjacent topics broadens the "
                    "discovery surface without sacrificing the depth "
                    "boundary."
                ),
                confidence=0.5,
                target_field="source_config.researcher.research_topics",
            )
        )

    _emit_stage(
        f"reflection.researcher:end hunks_proposed={len(hunks)} "
        f"sections={sorted({h['section'] for h in hunks})}"
    )
    return hunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_records_with_decision(
    final_judgments: Iterable[dict],
    decisions: set[str],
) -> Iterable[dict]:
    for row in final_judgments:
        if not isinstance(row, dict):
            continue
        decision = (
            row.get("decision")
            or row.get("terminal_decision")
            or (row.get("full_decision") or {}).get("decision")
            or (row.get("facial_decision") or {}).get("decision")
        )
        if decision in decisions:
            yield row


def _candidate_record(row: dict) -> dict:
    """Pull the candidate record from any of the known shapes."""

    cand = row.get("candidate")
    if isinstance(cand, dict):
        return cand
    payload = row.get("terminal_payload") or {}
    if isinstance(payload, dict):
        cr = payload.get("candidate_record")
        if isinstance(cr, dict):
            return cr
    return {}


def _extract_country_code(affiliation: str) -> str | None:
    """Pull a parenthesized ISO country code from an affiliation string
    like ``"MIT (US)"``. Defensive against malformed input."""

    affiliation = affiliation.strip()
    if not affiliation.endswith(")") or "(" not in affiliation:
        return None
    candidate = affiliation.rsplit("(", 1)[1].rstrip(")").strip().upper()
    if len(candidate) == 2 and candidate.isalpha():
        return candidate
    return None


def _quartile_summary(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "median": sorted_values[len(sorted_values) // 2],
        "max": sorted_values[-1],
    }


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
    """Standard Gate-2 hunk dict shape (matches reflection.py propose-phase)."""

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


def _context_quality_label(narrative: dict[str, Any]) -> str:
    saves = narrative.get("totals", {}).get("saves_total", 0)
    if saves <= 0:
        return "empty"
    if saves < 5:
        return "thin"
    return "substantive"


def _load_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return list(read_jsonl(path))
    except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
        logger.warning(
            "researcher_reflection: failed to read %s (%s); treating as empty",
            path,
            exc,
        )
        return []
