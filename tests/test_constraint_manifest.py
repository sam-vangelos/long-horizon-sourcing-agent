"""P3b (Wave 2 slice 8): constraint manifest — owner / actuator / verifier per
constraint class, zero-owner stated constraints abort intake.

Guardrails from plans/sourcing-rigor-hardening.md P3: manifest schema test +
golden manifest fixture for a comp-band brief (aborts).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from shared.constraint_manifest import (
    ConstraintManifestError,
    aggregate_unsupported_dimensions,
    assert_constraint_manifest_runnable,
    build_constraint_manifest,
)


def _brief(**overrides) -> SimpleNamespace:
    base = {
        "permanent_filters": {},
        "intake_notes": "",
        "instructions": [],
        "employer_blacklist": [],
        "experience_floor": {},
        "raw": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_manifest_schema_covers_all_constraint_classes():
    manifest = build_constraint_manifest(_brief())
    assert manifest["schema_version"] == 1
    assert set(manifest["classes"]) == {
        "geography",
        "seniority",
        "employer_blacklist",
        "compensation",
    }
    for entry in manifest["classes"].values():
        assert set(entry) == {
            "stated_in_brief",
            "stated_value",
            "owner_layer",
            "actuator",
            "verify_method",
            "status",
        }
    assert manifest["requested_but_unsupported"] == {}


def test_geography_stated_is_owned_with_actuator_and_verifier():
    manifest = build_constraint_manifest(
        _brief(permanent_filters={"Location": "New York City Metropolitan Area"})
    )
    geography = manifest["classes"]["geography"]
    assert geography["stated_in_brief"] is True
    assert geography["status"] == "owned"
    assert geography["owner_layer"]
    assert geography["actuator"]
    assert geography["verify_method"]


def test_unstated_constraints_do_not_abort():
    manifest = build_constraint_manifest(_brief())
    assert all(
        entry["status"] == "unstated" for entry in manifest["classes"].values()
    )
    assert_constraint_manifest_runnable(manifest)  # must not raise


def test_comp_band_in_intake_notes_is_zero_owner_and_aborts():
    """The golden comp-band fixture: a recruiter-stated comp band has zero
    owners repo-wide, so intake aborts with the named error instead of
    silently dropping the constraint."""
    manifest = build_constraint_manifest(
        _brief(intake_notes="Target comp band is $180k - $220k base pay, do not exceed.")
    )
    compensation = manifest["classes"]["compensation"]
    assert compensation["stated_in_brief"] is True
    assert compensation["status"] == "zero_owner"
    assert "intake_notes" in compensation["stated_value"]

    with pytest.raises(ConstraintManifestError, match="compensation"):
        assert_constraint_manifest_runnable(manifest)


def test_comp_mention_in_instructions_aborts():
    manifest = build_constraint_manifest(
        _brief(instructions=["Only candidates within the stated salary range."])
    )
    assert manifest["classes"]["compensation"]["status"] == "zero_owner"
    with pytest.raises(ConstraintManifestError):
        assert_constraint_manifest_runnable(manifest)


def test_jd_style_comp_boilerplate_outside_command_surfaces_is_ignored():
    """role_description carries the JD body — "competitive salary" boilerplate
    there is not a recruiter constraint and must not abort every run."""
    manifest = build_constraint_manifest(
        _brief(role_description="We offer competitive salary and equity.")
    )
    assert manifest["classes"]["compensation"]["status"] == "unstated"
    assert_constraint_manifest_runnable(manifest)


def test_blacklist_and_seniority_statedness():
    manifest = build_constraint_manifest(
        _brief(
            employer_blacklist=["Acme Corp"],
            experience_floor={"minimum_years": 8},
        )
    )
    assert manifest["classes"]["employer_blacklist"]["status"] == "owned"
    assert "Acme Corp" in manifest["classes"]["employer_blacklist"]["stated_value"]
    assert manifest["classes"]["seniority"]["status"] == "owned"


def test_aggregate_unsupported_dimensions_counts_across_receipts():
    strings = [
        SimpleNamespace(
            surface_receipt={"unsupported_controls": ["seniority", "industries"]}
        ),
        SimpleNamespace(surface_receipt={"unsupported_controls": ["seniority"]}),
        SimpleNamespace(surface_receipt={}),
        SimpleNamespace(surface_receipt=None),
    ]
    assert aggregate_unsupported_dimensions(strings) == {
        "seniority": 2,
        "industries": 1,
    }
