"""Contract-guard for the recruiter-facing API the frontend hydrates against.

WHY THIS EXISTS: the frontend redesign (a separate workstream) hydrates the
four recruiter-facing endpoints below. Their RESPONSE SHAPES are therefore
load-bearing contracts, not implementation details — a backend change that
alters a field set or a ``slice`` version tag would silently break the
frontend's hydration (it renders empty, with no obvious backend cause).

This test pins those shapes so OUR suite goes red the instant any future
backend work (R-SIM, R5a/R5b, Refactor Y, ...) would break the contract —
catching it on our side before it reaches the frontend. It asserts the
SHAPE (the set of fields + the slice literal), never the VALUES (those move).

If a contract change is INTENTIONAL: update this test AND coordinate the
shape change with the frontend workstream in the same breath. A red here is
the reminder that the change is frontend-visible.

The frozen contracts (verified in code at write time):
  GET  /api/recruiter                       -> RecruiterResponse
  GET  /api/recruiter/{recruiter_id}/dashboard -> RecruiterDashboardResponse
  GET  /api/recruiter/preferences           -> RecruiterPreferencesResponse
  PUT  /api/recruiter/preferences           -> RecruiterPreferencesResponse
"""

from __future__ import annotations


def _field_names(model) -> set[str]:
    # pydantic v2: model_fields is the declared-field map.
    return set(model.model_fields.keys())


def _default_for(model, field: str):
    return model.model_fields[field].default


# --- GET /api/recruiter — the recruiter entity --------------------------------


def test_recruiter_entity_contract() -> None:
    from cloris.api._monolith import RecruiterResponse, RecruiterTasteSignalSummary

    assert _field_names(RecruiterResponse) == {
        "recruiter_id",
        "canonical_handle",
        "display_name",
        "briefs_count",
        "active_signals",
    }
    # active_signals elements are RecruiterTasteSignalSummary — pin that shape too.
    assert _field_names(RecruiterTasteSignalSummary) == {
        "id",
        "signal_kind",
        "domain",
        "source_brief_id",
        "confidence",
        "created_at",
        "payload",
    }


# --- GET /api/recruiter/{id}/dashboard — R3 persistence triad -----------------


def test_recruiter_dashboard_contract() -> None:
    from cloris.models import (
        RecruiterCalibrationEntry,
        RecruiterDashboardResponse,
        RecruiterPresenceEntry,
        RecruiterReflectionEntry,
    )

    assert _field_names(RecruiterDashboardResponse) == {
        "slice",
        "recruiter_id",
        "recruiter_handle",
        "presence",
        "calibration_drift",
        "reflection_trail",
    }
    # The slice version tag the frontend keys on. Bump = a frontend-coordinated change.
    assert _default_for(RecruiterDashboardResponse, "slice") == "v0-recruiter-slice-1"

    # PRESENCE: cross-brief accretion, NO verdict field (D-B — never flatten a
    # per-(person,role) verdict onto presence). Frozen at the type level.
    assert _field_names(RecruiterPresenceEntry) == {
        "person_id",
        "times_surfaced",
        "first_seen_brief",
        "last_seen_brief",
        "last_lifecycle_state",
    }
    assert "terminal_decision" not in _field_names(RecruiterPresenceEntry)
    assert "verdict" not in _field_names(RecruiterPresenceEntry)

    # CALIBRATION + REFLECTION entries (R3.2/R3.3) — the panels the frontend renders.
    assert _field_names(RecruiterCalibrationEntry) == {
        "domain",
        "drift",
        "brief_id",
        "total_markers",
        "by_marker_value",
        "by_capability_area",
        "weighted_markers_by_area",
        "source_state_dirs",
    }
    assert _field_names(RecruiterReflectionEntry) == {
        "reflection_id",
        "summary",
        "created_at",
        "brief_id",
        "current_phase",
        "started_at",
        "updated_at",
        "steering_iterations",
    }


# --- GET/PUT /api/recruiter/preferences ---------------------------------------


def test_recruiter_preferences_contract() -> None:
    from cloris.models import RecruiterPreferencesResponse

    assert _field_names(RecruiterPreferencesResponse) == {"slice", "preferences"}
    assert _default_for(RecruiterPreferencesResponse, "slice") == "v0-preferences-1"


# --- R-SIM S6: market-intel role_family must NOT leak into the recruiter API -


def test_role_family_does_not_leak_to_recruiter_api() -> None:
    # R-SIM (groundwork) adds MarketIdentity.role_family for cross-brief
    # market-intel warm-start. It is a MARKET-INTEL field and must stay
    # backend-private — it must NOT appear in any recruiter-facing response the
    # frontend hydrates. If a future change wires market-family data into the
    # recruiter API, this reddens and forces the frontend-contract conversation.
    from cloris.api._monolith import RecruiterResponse, RecruiterTasteSignalSummary
    from cloris.models import (
        RecruiterCalibrationEntry,
        RecruiterDashboardResponse,
        RecruiterPreferencesResponse,
        RecruiterPresenceEntry,
        RecruiterReflectionEntry,
    )

    recruiter_facing = [
        RecruiterResponse,
        RecruiterTasteSignalSummary,
        RecruiterDashboardResponse,
        RecruiterPresenceEntry,
        RecruiterCalibrationEntry,
        RecruiterReflectionEntry,
        RecruiterPreferencesResponse,
    ]
    for model in recruiter_facing:
        assert "role_family" not in _field_names(model), (
            f"{model.__name__} exposes role_family — market-intel field leaking "
            f"into the recruiter API the frontend hydrates"
        )
