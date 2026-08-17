"""Tests for `shared.confidentiality` — Executive Search Slice 1 scaffolding.

Pins the helper surface every aggregator/emitter will route through in
Slice 6:

- `aggregator_visibility(brief, surface_kind) -> "full" | "redacted" | "masked"`
- `should_emit_candidate_name(brief, artifact_kind) -> bool`
- `cross_brief_payload_filter(brief, payload) -> dict` (idempotent)

Slice 1 ships the helpers with no callers wired. Slice 6 wires call
sites and adds `tests/test_confidentiality_boundary.py` to enumerate
every aggregator and assert it routes through this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from shared.confidentiality import (
    BLIND_COUNT_MASK,
    BLIND_TITLE_MASK,
    ArtifactKind,
    ConfidentialityClass,
    SurfaceKind,
    aggregator_visibility,
    cross_brief_payload_filter,
    should_emit_candidate_name,
)


@dataclass
class _Brief:
    """Minimal brief-shaped object the helpers are meant to accept."""

    confidentiality_class: str = "open"


# ---------------------------------------------------------------------------
# ConfidentialityClass enum
# ---------------------------------------------------------------------------


def test_confidentiality_class_values() -> None:
    assert ConfidentialityClass.OPEN.value == "open"
    assert ConfidentialityClass.REFERENCEABLE.value == "referenceable"
    assert ConfidentialityClass.BLIND.value == "blind"


# ---------------------------------------------------------------------------
# aggregator_visibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface",
    [s for s in SurfaceKind],
)
def test_open_brief_is_full_on_every_surface(surface: SurfaceKind) -> None:
    """Open class — current behavior; everything visible everywhere."""

    brief = _Brief(confidentiality_class="open")
    assert aggregator_visibility(brief, surface) == "full"


@pytest.mark.parametrize(
    "surface,expected",
    [
        (SurfaceKind.BRIEF_AGGREGATOR, "full"),
        (SurfaceKind.STATUS_AGGREGATOR, "full"),
        (SurfaceKind.WORKSPACE, "full"),
        (SurfaceKind.CANDIDATE_DETAIL, "full"),
        (SurfaceKind.REFLECTION, "redacted"),
        (SurfaceKind.RUN_REPORT, "redacted"),
        (SurfaceKind.BRIEF_ITERATION, "redacted"),
        (SurfaceKind.MARKET_INTEL, "redacted"),
    ],
)
def test_referenceable_brief_visibility(
    surface: SurfaceKind, expected: str
) -> None:
    brief = _Brief(confidentiality_class="referenceable")
    assert aggregator_visibility(brief, surface) == expected


@pytest.mark.parametrize(
    "surface,expected",
    [
        (SurfaceKind.BRIEF_AGGREGATOR, "masked"),
        (SurfaceKind.STATUS_AGGREGATOR, "masked"),
        (SurfaceKind.REFLECTION, "masked"),
        (SurfaceKind.RUN_REPORT, "masked"),
        (SurfaceKind.BRIEF_ITERATION, "masked"),
        (SurfaceKind.MARKET_INTEL, "masked"),
        (SurfaceKind.WORKSPACE, "full"),
        (SurfaceKind.CANDIDATE_DETAIL, "full"),
    ],
)
def test_blind_brief_visibility(surface: SurfaceKind, expected: str) -> None:
    brief = _Brief(confidentiality_class="blind")
    assert aggregator_visibility(brief, surface) == expected


def test_aggregator_visibility_accepts_string_surface_kind() -> None:
    """Callers can pass either the enum or the string value."""

    brief = _Brief(confidentiality_class="blind")
    assert aggregator_visibility(brief, "reflection") == "masked"
    assert aggregator_visibility(brief, SurfaceKind.REFLECTION) == "masked"


def test_aggregator_visibility_rejects_unknown_surface_kind() -> None:
    brief = _Brief(confidentiality_class="open")
    with pytest.raises(ValueError, match="Unknown surface_kind"):
        aggregator_visibility(brief, "not_a_surface")


def test_aggregator_visibility_accepts_dict_brief() -> None:
    """V2-raw dicts work as well as Brief instances."""

    brief = {"confidentiality_class": "blind"}
    assert aggregator_visibility(brief, SurfaceKind.BRIEF_AGGREGATOR) == "masked"


def test_aggregator_visibility_defaults_to_open_when_class_missing() -> None:
    """Missing field → OPEN. Validation is the gate; the helper is fail-open."""

    brief = _Brief()  # default confidentiality_class="open"
    assert aggregator_visibility(brief, SurfaceKind.REFLECTION) == "full"


def test_aggregator_visibility_unknown_class_value_falls_back_to_open() -> None:
    """If validation lets a junk value through, the helper still defaults to OPEN.

    Per the module docstring: validate_v2_brief is the authoritative gate.
    """

    brief = _Brief(confidentiality_class="garbage")
    assert aggregator_visibility(brief, SurfaceKind.REFLECTION) == "full"


# ---------------------------------------------------------------------------
# should_emit_candidate_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_open_brief_emits_candidate_names_in_every_artifact(
    kind: ArtifactKind,
) -> None:
    brief = _Brief(confidentiality_class="open")
    assert should_emit_candidate_name(brief, kind) is True


@pytest.mark.parametrize(
    "kind",
    [ArtifactKind.WORKSPACE_CARD, ArtifactKind.CANDIDATE_DETAIL],
)
def test_referenceable_emits_names_inside_brief_surfaces(
    kind: ArtifactKind,
) -> None:
    brief = _Brief(confidentiality_class="referenceable")
    assert should_emit_candidate_name(brief, kind) is True


@pytest.mark.parametrize(
    "kind",
    [
        ArtifactKind.REFLECTION_PROSE,
        ArtifactKind.RUN_REPORT,
        ArtifactKind.BRIEF_ITERATION_DEBRIEF,
        ArtifactKind.MARKET_INTEL,
    ],
)
def test_referenceable_blocks_names_in_cross_surface_artifacts(
    kind: ArtifactKind,
) -> None:
    brief = _Brief(confidentiality_class="referenceable")
    assert should_emit_candidate_name(brief, kind) is False


@pytest.mark.parametrize(
    "kind",
    [ArtifactKind.WORKSPACE_CARD, ArtifactKind.CANDIDATE_DETAIL],
)
def test_blind_emits_names_inside_brief_surfaces(kind: ArtifactKind) -> None:
    """Blind class: in-brief surfaces still render names — recruiter does the job."""

    brief = _Brief(confidentiality_class="blind")
    assert should_emit_candidate_name(brief, kind) is True


@pytest.mark.parametrize(
    "kind",
    [
        ArtifactKind.REFLECTION_PROSE,
        ArtifactKind.RUN_REPORT,
        ArtifactKind.BRIEF_ITERATION_DEBRIEF,
        ArtifactKind.MARKET_INTEL,
    ],
)
def test_blind_blocks_names_in_cross_surface_artifacts(
    kind: ArtifactKind,
) -> None:
    brief = _Brief(confidentiality_class="blind")
    assert should_emit_candidate_name(brief, kind) is False


def test_should_emit_candidate_name_accepts_string_artifact_kind() -> None:
    brief = _Brief(confidentiality_class="referenceable")
    assert should_emit_candidate_name(brief, "reflection_prose") is False


def test_should_emit_candidate_name_rejects_unknown_artifact_kind() -> None:
    brief = _Brief(confidentiality_class="open")
    with pytest.raises(ValueError, match="Unknown artifact_kind"):
        should_emit_candidate_name(brief, "unknown")


# ---------------------------------------------------------------------------
# cross_brief_payload_filter
# ---------------------------------------------------------------------------


def test_cross_brief_filter_passes_through_open_brief() -> None:
    brief = _Brief(confidentiality_class="open")
    payload = {
        "role_title": "VP Engineering",
        "save_count": 12,
        "candidate_names": ["Jane Doe", "John Roe"],
    }
    assert cross_brief_payload_filter(brief, payload) == payload


def test_cross_brief_filter_strips_candidate_names_for_referenceable() -> None:
    brief = _Brief(confidentiality_class="referenceable")
    payload = {
        "role_title": "VP Engineering",
        "save_count": 12,
        "candidate_names": ["Jane Doe", "John Roe"],
        "save_reasons": ["..."],
    }
    out = cross_brief_payload_filter(brief, payload)
    assert "candidate_names" not in out
    assert "save_reasons" not in out
    assert out["role_title"] == "VP Engineering"
    assert out["save_count"] == 12


def test_cross_brief_filter_masks_title_and_count_for_blind() -> None:
    brief = _Brief(confidentiality_class="blind")
    payload = {
        "role_title": "VP Engineering",
        "save_count": 12,
        "candidate_names": ["Jane Doe"],
        "save_reasons": ["..."],
    }
    out = cross_brief_payload_filter(brief, payload)
    assert out["role_title"] == BLIND_TITLE_MASK
    assert out["save_count"] == BLIND_COUNT_MASK
    assert "candidate_names" not in out
    assert "save_reasons" not in out


def test_cross_brief_filter_is_idempotent() -> None:
    """filter(filter(p)) == filter(p) for any class."""

    payload = {
        "role_title": "VP Engineering",
        "save_count": 12,
        "candidate_names": ["Jane Doe"],
    }
    for cls in (
        ConfidentialityClass.OPEN,
        ConfidentialityClass.REFERENCEABLE,
        ConfidentialityClass.BLIND,
    ):
        brief = _Brief(confidentiality_class=cls.value)
        once = cross_brief_payload_filter(brief, payload)
        twice = cross_brief_payload_filter(brief, once)
        assert once == twice


def test_cross_brief_filter_does_not_mutate_input() -> None:
    brief = _Brief(confidentiality_class="blind")
    payload = {
        "role_title": "VP Engineering",
        "candidate_names": ["Jane Doe"],
    }
    snapshot = dict(payload)
    cross_brief_payload_filter(brief, payload)
    assert payload == snapshot


def test_cross_brief_filter_accepts_dict_brief() -> None:
    brief = {"confidentiality_class": "blind"}
    out = cross_brief_payload_filter(brief, {"role_title": "X", "save_count": 5})
    assert out["role_title"] == BLIND_TITLE_MASK
    assert out["save_count"] == BLIND_COUNT_MASK
