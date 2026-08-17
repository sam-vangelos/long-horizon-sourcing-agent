"""Calibration aggregator over ``candidates.judgment_accuracy``.

Phase 3.1 of the multi-agent execution plan
(``plans/multi-agent-execution-plan.md``). Read-only rollup over the
recruiter's calibration markers — the Phase C-bis Slice 0.5 columns
``judgment_accuracy`` and ``judgment_accuracy_at`` migrated in
``shared/runtime_state/store.py:381-390`` and surfaced on
``CandidateRecord`` at ``shared/runtime_state/read_models.py:227-228``.

Foundation-blocking for the threshold layer (Slice 3.2), the brief-patch
translator (Slice 3.3), and reflection ingestion (Slice 3.4).

## Read-only invariant

Opens the runtime-state SQLite via the URI ``mode=ro`` pattern, mirroring
``shared.runtime_state.read_models._open_readonly`` (read_models.py:277-302)
and ``cloris.control_plane``'s readers (control_plane.py:8-17). The
kernel rejects any DDL or INSERT, even if a future caller passes the
wrong path. A missing or unreadable DB collapses to an empty rollup so a
passive observer never crashes the chief-of-staff or reflection paths.

## Rollup shape

Returns counts per
``(capability_area, marker_value, confidence_quartile, terminal_decision)``
plus four per-axis convenience breakdowns. The full-key counts are the
canonical surface; the per-axis breakdowns exist so the threshold layer
(3.2) and translator (3.3) don't have to re-aggregate.

- ``capability_area`` is read from ``terminal_payload_json`` →
  ``full_decision.capability_area`` (the V2 wire shape used by LinkedIn /
  Researcher / GitHub full evaluations; see
  ``shared/judgment/templates.py:FullEvaluationResult.capability_area``).
  Pre-V2 LinkedIn rows that wrote ``OpusDecision.path`` instead are
  surfaced as ``None`` — the legacy ``path`` enum (``"pedigree" |
  "direct_experience" | "none"``) lives in a different namespace from
  V2 capability-area names and conflating them would feed the brief-
  patch translator (3.3) bad attributions. ``None`` is an explicit
  bucket so Slice 3.2 can decide whether to drop or surface
  unattributed volume.

- ``marker_value`` is the recruiter's ``judgment_accuracy`` enum value.
  Restricted to the writer-validated set
  (``store.py:660-666``); legacy or imported rows carrying any other
  value are dropped silently rather than skewing the rollup.

- ``confidence_quartile`` buckets confidence into static absolute bands:
  ``q1=[0, 0.25)``, ``q2=[0.25, 0.5)``, ``q3=[0.5, 0.75)``,
  ``q4=[0.75, 1.0]``, plus ``unknown`` when the payload doesn't carry
  a numeric confidence. Static (data-independent) bands keep the read
  stable across runs and avoid the small-N problem that data-relative
  quartiles introduce. The Slice 3.2 threshold layer cuts at raw
  ``confidence > 0.7`` (per execution-plan correction 3c); quartiles
  here are for downstream display rollups, not for thresholding.

- ``weighted_markers_by_area`` is the per-area confidence-weighted count
  the Slice 3.2 threshold layer (``market_intelligence/calibration_thresholds.py``)
  consumes. Each marker contributes 1; high-confidence
  (``confidence > HIGH_CONFIDENCE_THRESHOLD``) ``wrong`` / ``off_rubric``
  markers contribute 2 (per execution-plan correction 3c — when Cloris
  was sure and the recruiter said wrong, the signal is stronger). The
  aggregator owns this computation rather than the threshold layer
  because the row walk already has raw confidence in hand; surfacing
  raw confidence per row would force the threshold layer to either
  re-walk the DB or know the ``> 0.7`` cut. The weighted view is a
  lossless extension of ``by_capability_area``: the unweighted count is
  still available there.

- ``terminal_decision`` mirrors ``candidates.terminal_decision``. Carried
  through verbatim so the threshold layer can distinguish, e.g., a
  ``wrong`` marker on a ``SAVE`` (false positive) from a ``wrong`` marker
  on a ``REJECT`` (false negative).

## Out of scope for this slice

- Thresholding (Slice 3.2) — eligibility math lives downstream.
- Brief-patch translation (Slice 3.3) — the rollup is the input.
- Reflection ingestion (Slice 3.4) — caller wires this in.
- Designer per-principle markers (Slice 3.6) — those land in
  ``terminal_payload_json`` metadata; the aggregator only reads
  candidate-level ``judgment_accuracy`` today.
- Cross-source merging — the aggregator operates on a single per-source
  state SQLite. Cross-source briefs walk per-source DBs and merge in
  the consumer; keeping that boundary out of this primitive avoids
  coupling it to ``shared.output_paths`` resolvers.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from shared.runtime_state.read_models import (
    candidate_terminal_payload,
    extract_save_reason_and_confidence,
)


# Mirrors the writer-side enumeration at
# ``shared.runtime_state.store.set_candidate_judgment_accuracy``
# (store.py:660-666). The aggregator filters to this set so a future
# legacy/garbage value imported from older state DBs can't silently skew
# the rollup.
_ALLOWED_MARKER_VALUES: frozenset[str] = frozenset(
    {
        "useful",
        "wrong",
        "off_rubric",
        "overstated_depth",
        "understated_depth",
    }
)

QUARTILE_UNKNOWN: str = "unknown"
QUARTILE_LABELS: tuple[str, ...] = ("q1", "q2", "q3", "q4")

# Confidence cut applied by the Slice 3.2 threshold layer's weighted
# count. Lives here (not in ``calibration_thresholds.py``) because the
# aggregator must apply it during the row walk to surface
# ``weighted_markers_by_area``. The threshold module reads the
# precomputed weighted count and never sees raw confidence — keeping the
# wire surface narrow.
HIGH_CONFIDENCE_THRESHOLD: float = 0.7

# Marker values that earn the high-confidence weight bonus. Same intent
# as ``brief_polish.HALLUCINATION_OVERLAP_THRESHOLD``: a tuning starter
# that may move once telemetry shows the marker-mix distribution.
_HIGH_CONFIDENCE_BONUS_MARKERS: frozenset[str] = frozenset(
    {"wrong", "off_rubric"}
)
_HIGH_CONFIDENCE_WEIGHT: int = 2
_BASELINE_WEIGHT: int = 1


@dataclass(frozen=True)
class CalibrationRollupKey:
    """Composite key for the calibration rollup.

    Designed as a hashable frozen dataclass so callers can use it as a
    dict / Counter key directly. Field semantics live in the module
    docstring.
    """

    capability_area: str | None
    marker_value: str
    confidence_quartile: str
    terminal_decision: str | None


@dataclass(frozen=True)
class CalibrationRollup:
    """Structured rollup of ``judgment_accuracy`` markers for a brief.

    ``counts`` is the canonical full-key surface; the four per-axis
    mappings + ``weighted_markers_by_area`` are convenience views
    computed in the same pass so downstream slices don't re-aggregate.
    """

    brief_id: str
    source: str | None
    total_markers: int
    counts: Mapping[CalibrationRollupKey, int]
    by_marker_value: Mapping[str, int]
    by_capability_area: Mapping[str | None, int]
    by_confidence_quartile: Mapping[str, int]
    by_terminal_decision: Mapping[str | None, int]
    weighted_markers_by_area: Mapping[str | None, int]


@contextmanager
def _open_readonly(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    """Yield a read-only connection, or ``None`` for missing/unreadable DB.

    Local copy of the helper at
    ``shared.runtime_state.read_models._open_readonly`` (read_models.py:277).
    Replicated rather than imported because that helper is private and
    Phase 1.7 (``summarize_run_fn``) may concurrently extend
    ``read_models.py``; importing a private symbol would invite a merge
    collision while gaining nothing the local copy can't provide.
    """

    if not db_path.exists():
        yield None
        return
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        yield conn
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        yield None
    finally:
        if conn is not None:
            conn.close()


def confidence_quartile(confidence: float | None) -> str:
    """Bucket a confidence into the static absolute quartile band.

    Public so downstream slices (3.2 thresholding) can apply the same
    bucket boundaries when they need to project a fresh confidence value
    onto the same axis as the rollup.
    """

    if confidence is None:
        return QUARTILE_UNKNOWN
    if confidence < 0.25:
        return "q1"
    if confidence < 0.5:
        return "q2"
    if confidence < 0.75:
        return "q3"
    return "q4"


def _extract_capability_area(terminal_payload_json: str | None) -> str | None:
    """Pull ``capability_area`` from the canonical terminal payload shape.

    Reads ``payload['full_decision']['capability_area']`` only — the V2
    wire field. Pre-V2 LinkedIn rows that encoded an
    ``OpusDecision.path`` enum collapse to ``None`` here on purpose; see
    the module docstring for the rationale.
    """

    payload = candidate_terminal_payload(terminal_payload_json or "{}")
    if payload is None:
        return None
    full_decision = payload.get("full_decision")
    if isinstance(full_decision, dict):
        cap = full_decision.get("capability_area")
        if isinstance(cap, str) and cap.strip():
            return cap.strip()
    return None


def _extract_confidence(terminal_payload_json: str | None) -> float | None:
    """Confidence read via the canonical wire helper."""

    payload = candidate_terminal_payload(terminal_payload_json or "{}")
    _, confidence = extract_save_reason_and_confidence(payload)
    return confidence


def aggregate_calibration_markers(
    db_path: Path,
    *,
    brief_id: str,
    source: str | None = None,
) -> CalibrationRollup:
    """Aggregate ``judgment_accuracy`` markers across a brief's candidates.

    Pure read. Pass ``source`` to scope to one launcher source; omit
    to aggregate across all sources represented in this state DB.

    A missing DB file, an unreadable DB, or a pre-Phase-C-bis schema
    that lacks the ``judgment_accuracy`` column collapses to an empty
    rollup — same convention as other passive read helpers in
    ``shared.runtime_state.read_models``.
    """

    counts: Counter[CalibrationRollupKey] = Counter()
    by_marker_value: Counter[str] = Counter()
    by_capability_area: Counter[str | None] = Counter()
    by_confidence_quartile: Counter[str] = Counter()
    by_terminal_decision: Counter[str | None] = Counter()
    weighted_markers_by_area: Counter[str | None] = Counter()

    sql = (
        "SELECT judgment_accuracy, terminal_decision, terminal_payload_json "
        "FROM candidates "
        "WHERE brief_id = ? AND judgment_accuracy IS NOT NULL"
    )
    params: tuple = (brief_id,)
    if source is not None:
        sql += " AND source = ?"
        params = (brief_id, source)

    rows: list[sqlite3.Row] = []
    with _open_readonly(db_path) as conn:
        if conn is not None:
            try:
                rows = list(conn.execute(sql, params).fetchall())
            except sqlite3.OperationalError:
                # Pre-Phase-C-bis schema (no ``judgment_accuracy``
                # column). Bootstrap at store.py:381-390 adds it
                # idempotently, so any post-bootstrap DB has it; this
                # branch only fires against legacy snapshots.
                rows = []

    for row in rows:
        marker = row["judgment_accuracy"]
        if marker not in _ALLOWED_MARKER_VALUES:
            # Defensive: writer validates today (store.py:660-671), but
            # imported legacy rows might carry historical strings. Skip
            # rather than crash so one bad row doesn't poison the rollup.
            continue
        terminal_decision = row["terminal_decision"]
        payload_json = row["terminal_payload_json"]
        capability_area = _extract_capability_area(payload_json)
        confidence = _extract_confidence(payload_json)
        quartile = confidence_quartile(confidence)

        key = CalibrationRollupKey(
            capability_area=capability_area,
            marker_value=marker,
            confidence_quartile=quartile,
            terminal_decision=terminal_decision,
        )
        counts[key] += 1
        by_marker_value[marker] += 1
        by_capability_area[capability_area] += 1
        by_confidence_quartile[quartile] += 1
        by_terminal_decision[terminal_decision] += 1

        weight = _BASELINE_WEIGHT
        if (
            marker in _HIGH_CONFIDENCE_BONUS_MARKERS
            and confidence is not None
            and confidence > HIGH_CONFIDENCE_THRESHOLD
        ):
            weight = _HIGH_CONFIDENCE_WEIGHT
        weighted_markers_by_area[capability_area] += weight

    return CalibrationRollup(
        brief_id=brief_id,
        source=source,
        total_markers=sum(counts.values()),
        counts=dict(counts),
        by_marker_value=dict(by_marker_value),
        by_capability_area=dict(by_capability_area),
        by_confidence_quartile=dict(by_confidence_quartile),
        by_terminal_decision=dict(by_terminal_decision),
        weighted_markers_by_area=dict(weighted_markers_by_area),
    )


# ---------------------------------------------------------------------------
# Langfuse dataset export — Phase 2 of Langfuse adoption.
#
# Sibling read helper that walks the same ``candidates.judgment_accuracy``
# rows :func:`aggregate_calibration_markers` aggregates from, but emits
# one row per (candidate, marker, full_decision) tuple in the typed
# Langfuse dataset shape (``input`` / ``expected_output`` / ``metadata``)
# pinned in the plan body.
#
# Pure read. Same ``mode=ro`` defensive posture as the aggregator;
# doesn't push to Langfuse — the CLI tool at
# ``tools/sync_judgment_datasets.py`` consumes the returned list and
# drives the Langfuse API with batching + Retry-After respect.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LangfuseDatasetRow:
    """One row destined for a Langfuse dataset.

    Mirrors Langfuse's typed ``DatasetItem`` shape:
    ``input`` / ``expected_output`` / ``metadata`` are dicts. The
    schema is pinned in the Phase 2 plan body — every field listed
    there has a concrete source in the DB row OR the brief, except
    where v1 limitations (noted below) apply.
    """

    input: dict
    expected_output: dict
    metadata: dict

    # Idempotency anchor for the sync tool. The CLI dedupes by
    # ``(dataset_name, identity_key)`` so re-running the sync against
    # the same state-dir doesn't double-emit. Read directly from
    # ``candidates.identity_key`` so it survives schema migrations.
    identity_key: str

    def to_dataset_item(self) -> dict:
        """JSON-serializable shape ready for the Langfuse API."""

        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
        }


def build_langfuse_dataset_rows(
    db_path: Path,
    *,
    brief_id: str,
    brief_dict: dict | None = None,
    source: str | None = None,
) -> list[LangfuseDatasetRow]:
    """Walk ``candidates.judgment_accuracy`` rows and emit dataset rows.

    Phase 2 of Langfuse adoption. Pure read; mirrors the
    :func:`aggregate_calibration_markers` walk shape so the contract
    stays consistent across calibration consumers.

    Returns a list of :class:`LangfuseDatasetRow`. Empty when:
    - DB missing / unreadable / lacks ``judgment_accuracy`` column.
    - No rows match the ``brief_id`` filter (and optional ``source``).
    - All rows carry markers outside :data:`_ALLOWED_MARKER_VALUES`
      (defensive — same drop posture as the aggregator).

    ``brief_dict`` is the recruiter-authored brief content (loaded
    via ``shared.brief_loader.load_brief().raw`` or equivalent) used
    to populate the row's ``input.capability_areas`` and
    ``input.depth_distinction``. Pass ``None`` to skip the brief-
    derived fields and emit the row with empty placeholders — useful
    for unit tests.
    """

    sql = (
        "SELECT c.identity_key, c.source, c.judgment_accuracy, "
        "judgment_accuracy_at, terminal_decision, "
        "terminal_payload_json, display_name, profile_url, "
        "ca.payload_json AS attempt_payload_json "
        "FROM candidates c "
        "LEFT JOIN candidate_attempts ca ON ca.id = c.last_attempt_id "
        "WHERE c.brief_id = ? AND c.judgment_accuracy IS NOT NULL"
    )
    params: tuple = (brief_id,)
    if source is not None:
        sql += " AND source = ?"
        params = (brief_id, source)

    rows: list[sqlite3.Row] = []
    with _open_readonly(db_path) as conn:
        if conn is not None:
            try:
                rows = list(conn.execute(sql, params).fetchall())
            except sqlite3.OperationalError:
                # Pre-Phase-C-bis schema; same posture as aggregator.
                rows = []

    capability_areas, depth_distinction = _brief_extract(brief_dict)
    trace_cache: dict[str, Any] = {}

    out: list[LangfuseDatasetRow] = []
    for row in rows:
        marker = row["judgment_accuracy"]
        if marker not in _ALLOWED_MARKER_VALUES:
            continue

        identity_key = row["identity_key"] or ""
        if not identity_key:
            # Defensive: identity_key is the idempotency anchor; a
            # missing one would let duplicate rows land on re-sync.
            continue

        payload_json = row["terminal_payload_json"] or "{}"
        rationale, confidence = _extract_rationale_and_confidence(payload_json)
        prompt_capture = _extract_prompt_capture(row["attempt_payload_json"])
        observability = _extract_terminal_observability(payload_json)
        candidate_source = (row["source"] or "").strip() or "unknown"
        display_name = (row["display_name"] or "").strip()
        profile_url = (row["profile_url"] or "").strip()

        # ``candidate_summary`` is the recruiter-readable string the
        # regression runner surfaces alongside the LLM's re-eval.
        # v1 shape: name + URL + rationale (the part the LLM
        # produced at eval time).
        summary_parts = []
        if display_name:
            summary_parts.append(display_name)
        if profile_url:
            summary_parts.append(profile_url)
        if rationale:
            summary_parts.append(rationale)
        candidate_summary = " — ".join(summary_parts) or "(no summary)"
        candidate_text = _candidate_text_for_row(
            prompt_capture=prompt_capture,
            candidate_summary=candidate_summary,
        )
        trace_id = _first_non_empty_str(
            prompt_capture.get("trace_id"),
            observability.get("trace_id"),
        )
        observation_id = _first_non_empty_str(
            prompt_capture.get("observation_id"),
            observability.get("observation_id"),
        )
        trace_url = _first_non_empty_str(
            prompt_capture.get("trace_url"),
            observability.get("trace_url"),
        )
        capture_mode = (
            "captured_prompt"
            if isinstance(prompt_capture.get("candidate_text"), str)
            and str(prompt_capture.get("candidate_text")).strip()
            else "legacy_summary_fallback"
        )
        cascade_route_hit = _resolve_cascade_route_hit(
            trace_id=trace_id,
            observation_id=observation_id,
            trace_cache=trace_cache,
        )

        out.append(
            LangfuseDatasetRow(
                input={
                    "brief_id": brief_id,
                    "candidate_summary": candidate_summary,
                    "candidate_text": candidate_text,
                    "capability_areas": list(capability_areas),
                    "depth_distinction": dict(depth_distinction),
                    "source": candidate_source,
                },
                expected_output={
                    "judgment_accuracy": marker,
                    "full_decision_rationale": rationale,
                    "recruiter_marker_set_at": (
                        row["judgment_accuracy_at"] or ""
                    ),
                },
                metadata={
                    "identity_key": identity_key,
                    "confidence_at_eval": (
                        float(confidence) if confidence is not None else 0.0
                    ),
                    "trace_id": trace_id,
                    "observation_id": observation_id,
                    "trace_url": trace_url,
                    "capture_mode": capture_mode,
                    "cascade_route_hit": cascade_route_hit,
                },
                identity_key=identity_key,
            )
        )

    return out


def _brief_extract(
    brief_dict: dict | None,
) -> tuple[list[dict], dict]:
    """Pull ``capability_areas`` + ``depth_distinction`` from a brief dict.

    Tolerant of a missing brief (returns empty placeholders) so unit
    tests can call the export helper without loading a real brief.
    """

    if not isinstance(brief_dict, dict):
        return [], {}
    raw_areas = brief_dict.get("capability_areas") or []
    capability_areas: list[dict] = []
    if isinstance(raw_areas, list):
        for area in raw_areas:
            if isinstance(area, dict):
                capability_areas.append(dict(area))
    depth = brief_dict.get("depth_distinction") or {}
    depth_distinction = dict(depth) if isinstance(depth, dict) else {}
    return capability_areas, depth_distinction


def _extract_rationale_and_confidence(
    terminal_payload_json: str,
) -> tuple[str, float | None]:
    """Read ``full_decision.rationale`` + ``confidence`` from the wire payload."""

    payload = candidate_terminal_payload(terminal_payload_json or "{}")
    rationale, confidence = extract_save_reason_and_confidence(payload)
    return (rationale or ""), confidence


def _safe_json_dict(raw: str | bytes | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _extract_prompt_capture(attempt_payload_json: str | bytes | None) -> dict[str, Any]:
    payload = _safe_json_dict(attempt_payload_json)
    prompt_capture = payload.get("prompt_capture")
    return dict(prompt_capture) if isinstance(prompt_capture, dict) else {}


def _extract_terminal_observability(
    terminal_payload_json: str | None,
) -> dict[str, Any]:
    payload = candidate_terminal_payload(terminal_payload_json or "{}")
    if not isinstance(payload, dict):
        return {}
    observability = payload.get("observability")
    return dict(observability) if isinstance(observability, dict) else {}


def _candidate_text_for_row(
    *,
    prompt_capture: dict[str, Any],
    candidate_summary: str,
) -> str:
    candidate_text = prompt_capture.get("candidate_text")
    if isinstance(candidate_text, str) and candidate_text.strip():
        return candidate_text
    return candidate_summary


def _first_non_empty_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _langfuse_trace_client() -> Any | None:
    try:
        from shared.observability import is_active
        from shared.observability.langfuse_client import get_client
    except ImportError:
        return None

    if not is_active():
        return None
    client = get_client()
    inner = getattr(client, "_inner", None)
    if inner is None:
        return None
    api = getattr(inner, "api", None)
    if api is None:
        return None
    trace_client = getattr(api, "trace", None)
    if trace_client is not None:
        return trace_client
    return getattr(api, "traces", None)


def _fetch_trace_details(trace_id: str) -> Any | None:
    trace_client = _langfuse_trace_client()
    if trace_client is None or not hasattr(trace_client, "get"):
        return None
    try:
        return trace_client.get(trace_id, fields="observations")
    except Exception:
        return None


def _trace_observations(trace_details: Any) -> list[Any]:
    if isinstance(trace_details, dict):
        observations = trace_details.get("observations")
    else:
        observations = getattr(trace_details, "observations", None)
    return list(observations) if isinstance(observations, (list, tuple)) else []


def _observation_id(observation: Any) -> str | None:
    if isinstance(observation, dict):
        value = observation.get("id")
    else:
        value = getattr(observation, "id", None)
    return value if isinstance(value, str) and value else None


def _observation_metadata(observation: Any) -> dict[str, Any]:
    if isinstance(observation, dict):
        metadata = observation.get("metadata")
    else:
        metadata = getattr(observation, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _select_trace_observation(
    observations: list[Any],
    observation_id: str | None,
) -> Any | None:
    if observation_id:
        for observation in observations:
            if _observation_id(observation) == observation_id:
                return observation
    return observations[0] if observations else None


def _resolve_cascade_route_hit(
    *,
    trace_id: str | None,
    observation_id: str | None,
    trace_cache: dict[str, Any],
) -> str | None:
    if not trace_id:
        return None
    if trace_id not in trace_cache:
        trace_cache[trace_id] = _fetch_trace_details(trace_id)
    trace_details = trace_cache.get(trace_id)
    if trace_details is None:
        return None
    observation = _select_trace_observation(
        _trace_observations(trace_details),
        observation_id,
    )
    metadata = _observation_metadata(observation) if observation is not None else {}
    fallback_reason = metadata.get("cascade.fallback_reason")
    if isinstance(fallback_reason, str) and fallback_reason.strip():
        return fallback_reason
    return "clean"


__all__ = [
    "CalibrationRollup",
    "CalibrationRollupKey",
    "HIGH_CONFIDENCE_THRESHOLD",
    "LangfuseDatasetRow",
    "QUARTILE_LABELS",
    "QUARTILE_UNKNOWN",
    "aggregate_calibration_markers",
    "build_langfuse_dataset_rows",
    "confidence_quartile",
]
