"""Confidentiality contract for executive-search briefs.

Slice 1 of the executive-search module ships this module as scaffolding
only — Slice 6 wires the helpers into every aggregator and emitter that
crosses a brief boundary (`/api/briefs`, `/api/status`, reflection polish,
run reports, `tools/iterate_brief.py`).

The contract is product-level, not security-level: it gates what the
recruiter sees on Cloris's surfaces. It does not defend against an
attacker with API access (Cloris is single-user-desktop today; the
auth layer is a follow-up surfaced in the spec's "Follow-ups" section).

Three classes; one read path:

- ``open`` — current behavior; everything visible everywhere.
- ``referenceable`` — title + role visible in cross-brief aggregators;
  candidate names visible inside the brief's own workspace + candidate
  detail surfaces, but NOT in cross-surface emissions (reflection,
  run report, brief-iteration debrief, market intel).
- ``blind`` — title masked in cross-brief aggregators; save count
  masked; candidate names invisible across any cross-surface emission.
  In-brief workspace + candidate-detail surfaces still render names so
  the recruiter can do their job.

Single source of truth: every aggregator/emitter calls into this module.
A regression test in Slice 6 (``tests/test_confidentiality_boundary.py``)
enumerates the call sites and asserts each one routes through these
helpers — that is the leak guard.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ConfidentialityClass(str, Enum):
    """Three confidentiality postures a brief can declare.

    Stored on ``Brief.confidentiality_class`` and on V2 raw under
    ``confidentiality_class``. Default is :attr:`OPEN`.
    """

    OPEN = "open"
    REFERENCEABLE = "referenceable"
    BLIND = "blind"


class SurfaceKind(str, Enum):
    """Where a brief is being read from. Used by aggregator visibility.

    Cross-brief surfaces aggregate or emit across multiple briefs and
    are subject to redaction/masking. Per-brief surfaces are inside
    the brief's own scope and always render the recruiter's full view.
    """

    # Cross-brief surfaces — gated.
    BRIEF_AGGREGATOR = "brief_aggregator"
    STATUS_AGGREGATOR = "status_aggregator"
    REFLECTION = "reflection"
    RUN_REPORT = "run_report"
    BRIEF_ITERATION = "brief_iteration"
    MARKET_INTEL = "market_intel"

    # Per-brief surfaces — always full.
    WORKSPACE = "workspace"
    CANDIDATE_DETAIL = "candidate_detail"


class ArtifactKind(str, Enum):
    """What kind of artifact is being emitted. Used by name-emission gating.

    A subset of :class:`SurfaceKind` framed from the emitter's side: a
    candidate name being placed inside an artifact, not a surface
    rendering a brief summary.
    """

    WORKSPACE_CARD = "workspace_card"
    CANDIDATE_DETAIL = "candidate_detail"
    REFLECTION_PROSE = "reflection_prose"
    RUN_REPORT = "run_report"
    BRIEF_ITERATION_DEBRIEF = "brief_iteration_debrief"
    MARKET_INTEL = "market_intel"


# Default mask string for blind-class brief titles in cross-brief
# aggregators. The spec example "Confidential search — Sector A
# executive" sketches a future enhancement (sector inference); v1
# ships the generic placeholder.
BLIND_TITLE_MASK = "Confidential search"

# Default mask token for blind-class save counts in cross-brief
# aggregators. Em-dash, recruiter-facing.
BLIND_COUNT_MASK = "\u2014"


# Per-brief surfaces are always full. Cross-brief surfaces are gated by
# class. Indexed by (class, surface) → "full" | "redacted" | "masked".
_AGGREGATOR_VISIBILITY: dict[
    tuple[ConfidentialityClass, SurfaceKind], str
] = {
    # Open: full everywhere.
    **{(ConfidentialityClass.OPEN, surface): "full" for surface in SurfaceKind},
    # Referenceable: title + role visible cross-brief; names redacted
    # in cross-surface emissions; in-brief surfaces are full.
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.BRIEF_AGGREGATOR): "full",
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.STATUS_AGGREGATOR): "full",
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.REFLECTION): "redacted",
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.RUN_REPORT): "redacted",
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.BRIEF_ITERATION): "redacted",
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.MARKET_INTEL): "redacted",
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.WORKSPACE): "full",
    (ConfidentialityClass.REFERENCEABLE, SurfaceKind.CANDIDATE_DETAIL): "full",
    # Blind: title masked cross-brief; in-brief surfaces are full so the
    # recruiter can still do their job.
    (ConfidentialityClass.BLIND, SurfaceKind.BRIEF_AGGREGATOR): "masked",
    (ConfidentialityClass.BLIND, SurfaceKind.STATUS_AGGREGATOR): "masked",
    (ConfidentialityClass.BLIND, SurfaceKind.REFLECTION): "masked",
    (ConfidentialityClass.BLIND, SurfaceKind.RUN_REPORT): "masked",
    (ConfidentialityClass.BLIND, SurfaceKind.BRIEF_ITERATION): "masked",
    (ConfidentialityClass.BLIND, SurfaceKind.MARKET_INTEL): "masked",
    (ConfidentialityClass.BLIND, SurfaceKind.WORKSPACE): "full",
    (ConfidentialityClass.BLIND, SurfaceKind.CANDIDATE_DETAIL): "full",
}


# In-brief artifact kinds always emit candidate names. Cross-surface
# artifact kinds gate on confidentiality class.
_NAME_EMISSION: dict[
    tuple[ConfidentialityClass, ArtifactKind], bool
] = {
    **{
        (ConfidentialityClass.OPEN, kind): True for kind in ArtifactKind
    },
    (ConfidentialityClass.REFERENCEABLE, ArtifactKind.WORKSPACE_CARD): True,
    (ConfidentialityClass.REFERENCEABLE, ArtifactKind.CANDIDATE_DETAIL): True,
    (ConfidentialityClass.REFERENCEABLE, ArtifactKind.REFLECTION_PROSE): False,
    (ConfidentialityClass.REFERENCEABLE, ArtifactKind.RUN_REPORT): False,
    (ConfidentialityClass.REFERENCEABLE, ArtifactKind.BRIEF_ITERATION_DEBRIEF): False,
    (ConfidentialityClass.REFERENCEABLE, ArtifactKind.MARKET_INTEL): False,
    (ConfidentialityClass.BLIND, ArtifactKind.WORKSPACE_CARD): True,
    (ConfidentialityClass.BLIND, ArtifactKind.CANDIDATE_DETAIL): True,
    (ConfidentialityClass.BLIND, ArtifactKind.REFLECTION_PROSE): False,
    (ConfidentialityClass.BLIND, ArtifactKind.RUN_REPORT): False,
    (ConfidentialityClass.BLIND, ArtifactKind.BRIEF_ITERATION_DEBRIEF): False,
    (ConfidentialityClass.BLIND, ArtifactKind.MARKET_INTEL): False,
}


def _resolve_class(brief: Any) -> ConfidentialityClass:
    """Read ``confidentiality_class`` from a Brief dataclass or V2 dict.

    Tolerant of:
    - missing field (defaults to ``OPEN``)
    - ``None`` value (defaults to ``OPEN``)
    - unknown string values (defaults to ``OPEN``; ``validate_v2_brief``
      is the authoritative gate that prevents unknown values from
      reaching this code path in practice)
    """

    if isinstance(brief, Mapping):
        raw_value = brief.get("confidentiality_class", "open")
    else:
        raw_value = getattr(brief, "confidentiality_class", "open")
    if raw_value is None or raw_value == "":
        return ConfidentialityClass.OPEN
    if isinstance(raw_value, ConfidentialityClass):
        return raw_value
    try:
        return ConfidentialityClass(str(raw_value))
    except ValueError:
        return ConfidentialityClass.OPEN


def aggregator_visibility(brief: Any, surface_kind: SurfaceKind | str) -> str:
    """Return the visibility level a surface has on a brief.

    Returns one of ``"full"`` / ``"redacted"`` / ``"masked"``:

    - ``full`` — render brief title, role, candidate count, and any
      candidate names normally.
    - ``redacted`` — render brief title and role; do NOT emit
      candidate names or save reasons. Save count remains visible.
    - ``masked`` — replace brief title with :data:`BLIND_TITLE_MASK`
      and save count with :data:`BLIND_COUNT_MASK`. Do NOT emit any
      candidate-bearing detail.
    """

    if not isinstance(surface_kind, SurfaceKind):
        try:
            surface_kind = SurfaceKind(surface_kind)
        except ValueError as exc:
            raise ValueError(
                f"Unknown surface_kind {surface_kind!r}; expected one of "
                f"{[s.value for s in SurfaceKind]}"
            ) from exc

    cls = _resolve_class(brief)
    return _AGGREGATOR_VISIBILITY[(cls, surface_kind)]


def should_emit_candidate_name(
    brief: Any, artifact_kind: ArtifactKind | str
) -> bool:
    """Whether to emit a candidate name into an artifact.

    Returns ``True`` for in-brief surfaces regardless of class, and for
    cross-surface artifacts only when the brief is :attr:`ConfidentialityClass.OPEN`.
    """

    if not isinstance(artifact_kind, ArtifactKind):
        try:
            artifact_kind = ArtifactKind(artifact_kind)
        except ValueError as exc:
            raise ValueError(
                f"Unknown artifact_kind {artifact_kind!r}; expected one of "
                f"{[k.value for k in ArtifactKind]}"
            ) from exc

    cls = _resolve_class(brief)
    return _NAME_EMISSION[(cls, artifact_kind)]


# Keys that always carry candidate-bearing detail and should be stripped
# at any cross-brief boundary for non-OPEN classes. Slice 6 may extend
# this set as call sites surface new payload shapes.
_REDACTED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "candidate_names",
        "candidate_name",
        "save_reasons",
        "save_reason",
        "saves",
        "saved_candidates",
        "names",
    }
)

# Keys that carry brief identity (title, role) and should be masked
# specifically for BLIND class at cross-brief boundaries.
_MASKED_TITLE_KEYS: frozenset[str] = frozenset(
    {
        "role_title",
        "title",
        "brief_title",
    }
)

_MASKED_COUNT_KEYS: frozenset[str] = frozenset(
    {
        "save_count",
        "saves_count",
        "total_saves",
    }
)


def cross_brief_payload_filter(
    brief: Any, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Filter a payload before it crosses a brief boundary.

    Idempotent: ``filter(filter(p)) == filter(p)`` for any class.

    - ``open``: passthrough.
    - ``referenceable``: drop candidate-bearing keys; preserve title +
      role + count so cross-brief listings still surface the brief.
    - ``blind``: drop candidate-bearing keys, mask title, mask count.
    """

    cls = _resolve_class(brief)
    if cls is ConfidentialityClass.OPEN:
        return dict(payload)

    filtered: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _REDACTED_PAYLOAD_KEYS:
            continue
        if cls is ConfidentialityClass.BLIND:
            if key in _MASKED_TITLE_KEYS:
                filtered[key] = BLIND_TITLE_MASK
                continue
            if key in _MASKED_COUNT_KEYS:
                filtered[key] = BLIND_COUNT_MASK
                continue
        filtered[key] = value
    return filtered


__all__ = (
    "ArtifactKind",
    "BLIND_COUNT_MASK",
    "BLIND_TITLE_MASK",
    "ConfidentialityClass",
    "SurfaceKind",
    "aggregator_visibility",
    "cross_brief_payload_filter",
    "should_emit_candidate_name",
)
