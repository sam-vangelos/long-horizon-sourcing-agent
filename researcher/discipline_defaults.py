"""Per-discipline floor defaults + layered floor resolver — Slice 5/7.

Per Researcher Module Spec Opinion 7 (recruiter never types a floor at
intake): a layered priority resolves the deterministic-gate floors:

  1. Universal minimum — always applies (h_index ≥ 3, papers_in_window
     ≥ 1 in 36mo). Excludes zero-publication candidates only.
  2. Discipline default — when the recruiter picks a discipline in the
     wizard, the matching default overrides the universal minimum.
  3. Explicit override — when the recruiter (power-user) sets one of
     `h_index_floor`, `papers_in_window_floor`, `papers_in_window_months`
     directly in the brief, that value wins over the discipline default.

This module is the SINGLE place that knows the layered priority. The
gate sites at Slice 5 just call :func:`resolve_floors`.

Discipline values are first-pass calibration targets per Failure Mode 8
in the spec; values are intended to be edited based on post-trial
telemetry (facial-no rate per discipline). Don't over-engineer the
resolution layer before we have data.
"""

from __future__ import annotations

from typing import Any


# Universal minimum — Spec Opinion 7. Excludes zero-publication
# candidates only; the deterministic gate's job is budget control on
# obvious garbage, not editorial selection.
UNIVERSAL_MINIMUM: dict[str, int] = {
    "h_index_floor": 3,
    "papers_in_window_floor": 1,
    "papers_in_window_months": 36,
}


# Discipline defaults — first-pass values per Spec Opinion 7. Initial
# values per the spec body (Slice 7):
#   nlp        → h≥10, p≥3 in 24mo
#   ml_general → h≥8,  p≥3 in 24mo
#   vision     → h≥9,  p≥3 in 24mo
#   rl         → h≥7,  p≥3 in 24mo
#   systems    → h≥6,  p≥2 in 36mo
#   theory     → h≥5,  p≥2 in 48mo  (theory cites less, slower)
#   biomedical → h≥12, p≥4 in 24mo
#   other      → universal (no override)
DISCIPLINE_DEFAULTS: dict[str, dict[str, int]] = {
    "nlp": {"h_index_floor": 10, "papers_in_window_floor": 3, "papers_in_window_months": 24},
    "ml_general": {"h_index_floor": 8, "papers_in_window_floor": 3, "papers_in_window_months": 24},
    "vision": {"h_index_floor": 9, "papers_in_window_floor": 3, "papers_in_window_months": 24},
    "rl": {"h_index_floor": 7, "papers_in_window_floor": 3, "papers_in_window_months": 24},
    "systems": {"h_index_floor": 6, "papers_in_window_floor": 2, "papers_in_window_months": 36},
    "theory": {"h_index_floor": 5, "papers_in_window_floor": 2, "papers_in_window_months": 48},
    "biomedical": {"h_index_floor": 12, "papers_in_window_floor": 4, "papers_in_window_months": 24},
    # `other` falls back to UNIVERSAL_MINIMUM via the resolver — we
    # intentionally do NOT list it here so the resolver's empty-lookup
    # branch fires.
}


RECOGNIZED_DISCIPLINES: frozenset[str] = frozenset(
    list(DISCIPLINE_DEFAULTS.keys()) + ["other"]
)


def resolve_floors(source_config_researcher: dict[str, Any] | None) -> dict[str, int]:
    """Return the resolved floor dict per the layered priority.

    Returns ``{h_index_floor, papers_in_window_floor, papers_in_window_months}``
    populated per Spec Opinion 7 (explicit override → discipline default
    → universal minimum).

    ``source_config_researcher`` is the per-source sub-dict from the
    brief (i.e., ``brief.source_config["researcher"]``). Missing or
    non-dict ⇒ universal minimum.
    """

    config = source_config_researcher if isinstance(source_config_researcher, dict) else {}

    # Start from the universal minimum.
    resolved = dict(UNIVERSAL_MINIMUM)

    # Apply discipline default if the recruiter picked one.
    discipline = _normalize_discipline(config.get("discipline"))
    if discipline and discipline in DISCIPLINE_DEFAULTS:
        resolved.update(DISCIPLINE_DEFAULTS[discipline])

    # Apply explicit overrides last (power-user path).
    for key in ("h_index_floor", "papers_in_window_floor", "papers_in_window_months"):
        value = config.get(key)
        if isinstance(value, int) and value >= 0:
            resolved[key] = value
        elif isinstance(value, float) and value >= 0:
            resolved[key] = int(value)

    return resolved


def discipline_label(discipline: str) -> str:
    """Recruiter-facing label for a discipline key.

    Used by the facial fast-exit rationale builder to render
    "based on NLP defaults" rather than "based on nlp_default" leaks.
    """

    normalized = _normalize_discipline(discipline)
    return _DISCIPLINE_LABELS.get(normalized, "field defaults")


_DISCIPLINE_LABELS: dict[str, str] = {
    "nlp": "NLP",
    "ml_general": "general ML",
    "vision": "vision",
    "rl": "RL",
    "systems": "ML systems",
    "theory": "theory",
    "biomedical": "biomedical research",
    "other": "field defaults",
}


def _normalize_discipline(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()
