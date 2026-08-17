"""Researcher Slice 5/7 — discipline defaults + layered floor resolver.

Pins Spec Opinion 7's layered priority:
  explicit override → discipline default → universal minimum.

The resolver is the SINGLE place that knows the layering — gate sites
(Slice 5) and brief polish (Slice 7) both call resolve_floors and read
the resulting dict.
"""

from __future__ import annotations

from researcher.discipline_defaults import (
    DISCIPLINE_DEFAULTS,
    RECOGNIZED_DISCIPLINES,
    UNIVERSAL_MINIMUM,
    discipline_label,
    resolve_floors,
)


def test_universal_minimum_constants_match_spec_opinion_7() -> None:
    assert UNIVERSAL_MINIMUM == {
        "h_index_floor": 3,
        "papers_in_window_floor": 1,
        "papers_in_window_months": 36,
    }


def test_discipline_defaults_cover_seven_named_disciplines() -> None:
    """`other` falls back to UNIVERSAL_MINIMUM via the resolver — it's
    intentionally absent from DISCIPLINE_DEFAULTS so the empty-lookup
    branch fires."""

    expected_keys = {"nlp", "ml_general", "vision", "rl", "systems", "theory", "biomedical"}
    assert set(DISCIPLINE_DEFAULTS.keys()) == expected_keys


def test_recognized_disciplines_includes_other() -> None:
    assert "other" in RECOGNIZED_DISCIPLINES
    assert RECOGNIZED_DISCIPLINES == frozenset(
        list(DISCIPLINE_DEFAULTS.keys()) + ["other"]
    )


def test_resolve_floors_empty_config_returns_universal_minimum() -> None:
    assert resolve_floors({}) == UNIVERSAL_MINIMUM


def test_resolve_floors_none_config_returns_universal_minimum() -> None:
    assert resolve_floors(None) == UNIVERSAL_MINIMUM


def test_resolve_floors_discipline_overrides_universal() -> None:
    result = resolve_floors({"discipline": "nlp"})
    assert result == {
        "h_index_floor": 10,
        "papers_in_window_floor": 3,
        "papers_in_window_months": 24,
    }


def test_resolve_floors_explicit_override_wins_over_discipline() -> None:
    result = resolve_floors(
        {
            "discipline": "nlp",
            "h_index_floor": 15,  # power-user override
        }
    )
    assert result["h_index_floor"] == 15
    # The non-overridden keys still come from the discipline default.
    assert result["papers_in_window_floor"] == 3
    assert result["papers_in_window_months"] == 24


def test_resolve_floors_explicit_override_wins_over_universal_when_no_discipline() -> None:
    result = resolve_floors({"papers_in_window_floor": 5})
    assert result["papers_in_window_floor"] == 5
    assert result["h_index_floor"] == UNIVERSAL_MINIMUM["h_index_floor"]


def test_resolve_floors_other_discipline_falls_back_to_universal() -> None:
    """The `other` value is recognized but not in DISCIPLINE_DEFAULTS;
    it explicitly falls back to UNIVERSAL_MINIMUM."""

    assert resolve_floors({"discipline": "other"}) == UNIVERSAL_MINIMUM


def test_resolve_floors_unknown_discipline_falls_back_to_universal() -> None:
    """A typo / unknown discipline silently falls back to universal —
    we don't fail closed because the recruiter shouldn't be punished
    for a wizard-side typo."""

    assert resolve_floors({"discipline": "underwater_basket_weaving"}) == UNIVERSAL_MINIMUM


def test_resolve_floors_normalizes_discipline_case() -> None:
    """Case-insensitive discipline lookup."""

    assert resolve_floors({"discipline": "NLP"}) == DISCIPLINE_DEFAULTS["nlp"]
    assert resolve_floors({"discipline": "  Vision  "}) == DISCIPLINE_DEFAULTS["vision"]


def test_resolve_floors_accepts_float_overrides() -> None:
    """Brief polish can produce float-typed integer overrides (JSON
    parses 5.0 as float). Resolver should coerce to int."""

    result = resolve_floors({"h_index_floor": 7.0})
    assert result["h_index_floor"] == 7
    assert isinstance(result["h_index_floor"], int)


def test_resolve_floors_rejects_negative_overrides() -> None:
    """Negative floors don't take effect; the universal minimum (or
    discipline default) stays.
    """

    result = resolve_floors({"h_index_floor": -1})
    assert result["h_index_floor"] == UNIVERSAL_MINIMUM["h_index_floor"]


def test_discipline_label_returns_recruiter_readable_text() -> None:
    """Labels avoid the underscore-snake-case the spec calls out as
    engineer-vocab leak."""

    assert discipline_label("nlp") == "NLP"
    assert discipline_label("ml_general") == "general ML"
    assert discipline_label("biomedical") == "biomedical research"
    # Unknown / empty disciplines render as "field defaults" so the
    # fast-exit copy still reads cleanly.
    assert discipline_label("") == "field defaults"
    assert discipline_label("unknown") == "field defaults"
