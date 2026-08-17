"""Chief-of-staff handoff payloads — audit Move #1.

Closes the highest-blast-radius "Thing You're Not Seeing" finding from
the production-readiness audit: ``chief_of_staff_runs.handoff_payloads_json``
was declared at the schema layer (Slice 2.3) and persisted as ``{}``
forever. Multi-module runs read as three independent evals stitched
together rather than a coordinated team.

This module ships the small-version of the handoff arc — what the
schema, writer, and synthesis prompt all already supported but no
caller actually populated:

- :class:`HandoffPayload` — structured per-source summary written by
  each contributing module's run-end path.
- :func:`build_handoff_payload_from_evidence_batch` — pure transform
  reading a :class:`MarketEvidenceBatch` and returning the structured
  payload. Reflection-time consumers (the chief-of-staff synthesis
  call site at :mod:`market_intelligence.reflection`) build payloads
  for every contributing source, persist them via
  :meth:`shared.runtime_state.orchestration_store.OrchestrationStateStore.merge_handoff_payload`,
  and fold them into the synthesis user prompt.
- :func:`compose_handoff_context` — narrative composer for the
  synthesis prompt. Per-source summaries get rendered as a small JSON
  block so the LLM has every value to cite without re-deriving.

NOT the full broker arc: per-specialist fine-grained addressable
candidate-judging endpoints stay deferred to year-two (no registry
slot reserved for them today — P10 deleted the unpopulated
``judge_candidate_fn`` placeholder as dead theater; a real broker
arc adds its own slot when the demand materializes). This module's
payload shape is the substrate the broker arc will extend, not
replace.

Posture: pure functions over already-extracted shapes. Never raises;
malformed inputs collapse to empty payloads so the wider Gate-2 flow
(reflection / synthesis) doesn't abort on a degenerate single-source
read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_intelligence.schema import MarketEvidenceBatch


# Cap on top_saves length so a multi-module run with hundreds of saves
# per source doesn't bloat the orchestration row's JSON. The synthesis
# prompt only needs a handful of representative narratives — the
# recruiter walks the per-source workspace cards for the long tail.
MAX_TOP_SAVES_PER_SOURCE = 5

# Cap on per-save role-fit narrative length. The synthesis prompt
# composer truncates to a recruiter-readable one-liner; the full
# rationale lives on the candidate's terminal_payload_json.
MAX_ROLE_FIT_NARRATIVE_CHARS = 240


@dataclass(frozen=True)
class HandoffPayload:
    """One module's structured handoff payload.

    Persisted into ``chief_of_staff_runs.handoff_payloads_json`` keyed
    by ``source``. Multi-module runs accumulate one payload per
    contributing source; the synthesis call composes them into a
    cross-source narrative.

    Fields:
    - ``source``: canonical lowercase source key
      (``"linkedin"`` / ``"github"`` / ``"researcher"`` / ``"designer"``
      / ``"exec_search"``).
    - ``top_saves``: up to :data:`MAX_TOP_SAVES_PER_SOURCE` recruiter-
      readable per-candidate summaries — ``{"candidate_id",
      "role_fit_narrative", "confidence"}``. Ordered by confidence
      descending.
    - ``per_source_signal_summary``: one-paragraph editorial read
      summarizing the run's shape (saves, save rate, most-prevalent
      capability area, edge cases observed). Plain English, no
      engineer vocabulary.
    - ``confidence``: programmatic 0.0-1.0 confidence in the
      summary. Heuristic: signal-density across candidate_count +
      save_count + edge cases. NOT LLM self-rating.
    - ``candidate_count``: number of candidates this source surfaced.
      Carried alongside the prose so the synthesis prompt can
      enforce containment (the synthesis-side
      ``_containment_check`` requires the paragraph to cite a
      specific value from the per-source signals).
    - ``save_count``: number of SAVE-class candidates this source
      surfaced. Same containment role.
    """

    source: str
    top_saves: list[dict] = field(default_factory=list)
    per_source_signal_summary: str = ""
    confidence: float = 0.0
    candidate_count: int = 0
    save_count: int = 0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "top_saves": list(self.top_saves),
            "per_source_signal_summary": self.per_source_signal_summary,
            "confidence": round(float(self.confidence), 3),
            "candidate_count": int(self.candidate_count),
            "save_count": int(self.save_count),
        }


def build_handoff_payload_from_evidence_batch(
    batch: "MarketEvidenceBatch",
) -> HandoffPayload | None:
    """Build one source's :class:`HandoffPayload` from its evidence batch.

    Returns ``None`` when the batch's source is empty or the batch
    surfaced zero candidates (a source with nothing to say has
    nothing to hand off; the synthesis path filters these out at
    :func:`market_intelligence.reflection._contributing_sources_count`).

    Top saves are read from ``batch.final_judgments`` filtered to the
    SAVE-class decisions, ordered by ``full_decision.confidence`` (or
    the row-level ``confidence`` field), capped at
    :data:`MAX_TOP_SAVES_PER_SOURCE`.

    The signal summary is composed deterministically from the
    metrics_summary aggregates so a heuristic-backend run still
    produces a non-empty summary; LLM enrichment is a follow-up that
    doesn't change the contract.
    """

    source = (batch.source or "").strip().lower()
    if not source:
        return None

    metrics = batch.metrics_summary or {}
    candidate_count = int(metrics.get("candidate_volume", 0) or 0)
    if candidate_count <= 0:
        return None

    save_count = int(metrics.get("saved", 0) or 0)
    top_saves = _extract_top_saves(batch.final_judgments or [])

    summary = _compose_signal_summary(
        source=source,
        candidate_count=candidate_count,
        save_count=save_count,
        top_saves=top_saves,
    )
    confidence = _signal_density_confidence(
        candidate_count=candidate_count,
        save_count=save_count,
        top_saves_count=len(top_saves),
    )

    return HandoffPayload(
        source=source,
        top_saves=top_saves,
        per_source_signal_summary=summary,
        confidence=confidence,
        candidate_count=candidate_count,
        save_count=save_count,
    )


def compose_handoff_context(
    handoff_payloads: dict[str, dict] | None,
) -> dict | None:
    """Compose persisted per-source handoff payloads into prompt context.

    Returns ``None`` when the input is empty or absent — callers (the
    synthesis prompt builder at
    :mod:`cloris.chief_of_staff.prompts`) treat ``None`` as "no
    cross-source handoff context" and the prompt skips the section.

    The returned dict is JSON-serializable and fits inside
    :func:`cloris.chief_of_staff.prompts.build_chief_of_staff_user_prompt`'s
    payload contract: per-source summaries are surfaced under their
    source keys, each carrying the fields of :class:`HandoffPayload`
    plus the source-key humanized display name.

    Engine-vocab tokens (``"forward_deployed_engineering"``) are NOT
    rewritten here — that's the synthesis cascade's job. The handoff
    context is the structured input; humanization happens inside the
    synthesis paragraph the LLM produces and the cascade gates check
    after the fact.
    """

    if not handoff_payloads:
        return None
    composed: dict[str, dict] = {}
    for source, raw in sorted(handoff_payloads.items()):
        if not isinstance(raw, dict):
            continue
        normalized = {
            "source": str(raw.get("source") or source),
            "candidate_count": int(raw.get("candidate_count", 0) or 0),
            "save_count": int(raw.get("save_count", 0) or 0),
            "confidence": round(float(raw.get("confidence", 0.0) or 0.0), 3),
            "per_source_signal_summary": str(
                raw.get("per_source_signal_summary") or ""
            ),
            "top_saves": _truncate_top_saves(raw.get("top_saves") or []),
        }
        composed[source] = normalized
    return composed or None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_SAVE_DECISIONS = frozenset(
    {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
)


def _extract_top_saves(final_judgments: list[dict]) -> list[dict]:
    """Pull SAVE-class rows + project to the top_saves dict shape."""

    saves: list[dict] = []
    for row in final_judgments:
        if not isinstance(row, dict):
            continue
        decision = (
            row.get("decision")
            or row.get("terminal_decision")
            or (row.get("full_decision") or {}).get("decision")
        )
        if decision not in _SAVE_DECISIONS:
            continue
        raw_confidence = row.get("confidence")
        if raw_confidence is None:
            raw_confidence = (row.get("full_decision") or {}).get("confidence")
        confidence = _coerce_confidence(raw_confidence)
        narrative = _normalize_narrative(
            row.get("rationale")
            or (row.get("full_decision") or {}).get("rationale")
            or row.get("narrative")
            or ""
        )
        candidate_id = row.get("candidate_id") or row.get("identity_key") or ""
        saves.append(
            {
                "candidate_id": str(candidate_id),
                "role_fit_narrative": narrative,
                "confidence": confidence,
            }
        )

    saves.sort(
        key=lambda s: (
            s.get("confidence") is None,
            -(s.get("confidence") or 0.0),
        )
    )
    return saves[:MAX_TOP_SAVES_PER_SOURCE]


def _truncate_top_saves(top_saves: list) -> list[dict]:
    """Defensive coercion when reading persisted JSON. Drops malformed."""

    out: list[dict] = []
    for item in top_saves[:MAX_TOP_SAVES_PER_SOURCE]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "role_fit_narrative": _normalize_narrative(
                    item.get("role_fit_narrative") or ""
                ),
                "confidence": _coerce_confidence(item.get("confidence")),
            }
        )
    return out


def _normalize_narrative(value: object) -> str:
    """Compress whitespace + truncate to keep the orchestration row small."""

    text = " ".join(str(value or "").split()).strip()
    if len(text) > MAX_ROLE_FIT_NARRATIVE_CHARS:
        text = text[: MAX_ROLE_FIT_NARRATIVE_CHARS - 1].rstrip() + "\u2026"
    return text


def _coerce_confidence(value: object) -> float | None:
    """Clamped confidence, or None for missing / non-numeric values.

    P6 (Wave 2): a judge whose CONFIDENCE line failed to parse stores null —
    coercing that to 0.0 fabricated a stated-zero-confidence save on the
    Chief-of-Staff daily-brief surface, indistinguishable from a genuinely
    low-confidence one (correctness lens, slice 10). None stays None; the
    sort places unknown-confidence saves after measured ones.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        confidence = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return round(confidence, 3)


def _signal_density_confidence(
    *,
    candidate_count: int,
    save_count: int,
    top_saves_count: int,
) -> float:
    """Heuristic 0.0-1.0 confidence — programmatic, NOT LLM self-rated.

    Three signal channels:
    - candidate_count > 0 — module ran and surfaced something
    - save_count > 0 — module had a substantive read
    - top_saves with non-empty narratives — recruiter-citable detail

    Each channel contributes 1/3 weight when present. A module that
    surfaced 50 candidates with 0 saves still gets 1/3 (the negative
    read is informative). A module with 5 saves and rich narratives
    scores 1.0.
    """

    if candidate_count <= 0:
        return 0.0
    score = 1.0 / 3.0
    if save_count > 0:
        score += 1.0 / 3.0
    if top_saves_count > 0:
        score += 1.0 / 3.0
    return round(score, 3)


def _compose_signal_summary(
    *,
    source: str,
    candidate_count: int,
    save_count: int,
    top_saves: list[dict],
) -> str:
    """Deterministic editorial signal summary in plain English."""

    display = _humanize_source(source)
    if save_count == 0:
        return (
            f"{display} surfaced {candidate_count} candidates this run; "
            "none cleared the bar."
        )
    if save_count == 1:
        return (
            f"{display} surfaced {candidate_count} candidates and saved 1; "
            "a single substantive read."
        )
    return (
        f"{display} surfaced {candidate_count} candidates and saved {save_count}; "
        f"top {min(len(top_saves), MAX_TOP_SAVES_PER_SOURCE)} carry the strongest reads."
    )


_SOURCE_DISPLAY: dict[str, str] = {
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "researcher": "Researcher",
    "designer": "Designer",
    "exec_search": "Executive Search",
}


def _humanize_source(source: str) -> str:
    """Same shape as ``cloris.chief_of_staff.agent._humanize_source``."""

    raw = (source or "").strip().lower()
    if raw in _SOURCE_DISPLAY:
        return _SOURCE_DISPLAY[raw]
    if not raw:
        return ""
    return " ".join(part.capitalize() for part in raw.split("_") if part) or raw


__all__ = [
    "HandoffPayload",
    "MAX_ROLE_FIT_NARRATIVE_CHARS",
    "MAX_TOP_SAVES_PER_SOURCE",
    "build_handoff_payload_from_evidence_batch",
    "compose_handoff_context",
]
