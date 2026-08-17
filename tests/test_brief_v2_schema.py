"""Tests for the public V2 brief schema validation + legacy-merge surface
(``shared.brief_v2_schema``). Phase D Slice D-prep-C.

Pins the contract D2's brief detail/edit endpoint depends on:

- ``validate_v2_brief`` raises :class:`BriefSchemaError` with structured
  ``missing_keys`` / ``invalid_keys`` for invalid V2 input; passes silently
  for valid input.
- ``merge_legacy_brief`` splits a payload into V2 vs legacy parts per Fork
  B's "merge-with-deprecation" strategy. Deprecated keys are surfaced;
  unknown keys preserved with a separate flag.
- ``deprecation_message`` returns a Cloris-voice rationale for known
  deprecated keys.
"""

from __future__ import annotations

import pytest

from shared.brief_v2_schema import (
    BriefSchemaError,
    DEPRECATED_KEYS_BY_VERSION,
    MergedBrief,
    RECOGNIZED_CONFIDENTIALITY_CLASSES,
    RECOGNIZED_V2_KEYS,
    SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE,
    SOURCE_CONFIG_REQUIRED_KEYS_BY_SOURCE,
    deprecation_message,
    merge_legacy_brief,
    normalize_generated_engagement_context,
    validate_v2_brief,
)


def _minimal_valid_v2_brief() -> dict:
    return {
        "role_title": "Test Role",
        "capability_areas": [
            {
                "name": "Product engineering",
                "description": "Ships customer-facing systems end-to-end.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns architecture and ships.",
            "user_definition": "Maintains existing features.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }


# ---------------------------------------------------------------------------
# validate_v2_brief
# ---------------------------------------------------------------------------


def test_validate_v2_brief_accepts_minimal_valid_brief() -> None:
    validate_v2_brief(_minimal_valid_v2_brief())  # No raise.


def test_validate_v2_brief_accepts_candidate_register_terms_nested_key() -> None:
    payload = _minimal_valid_v2_brief()
    payload["capability_areas"][0]["candidate_register_terms"] = [
        "product engineering",
        "customer-facing systems",
    ]

    validate_v2_brief(payload)  # No raise; nested capability keys are open.


def test_experience_ceiling_fields_are_recognized_v2_keys() -> None:
    for key in (
        "maximum_years_experience",
        "experience_measure",
        "maximum_years_experience_is_hard",
    ):
        assert key in RECOGNIZED_V2_KEYS, f"{key!r} missing from RECOGNIZED_V2_KEYS"


def test_engagement_context_is_a_recognized_v2_key() -> None:
    assert "engagement_context" in RECOGNIZED_V2_KEYS


@pytest.mark.parametrize(
    "density,expected",
    [
        ("sparse", "coverage"),
        ("moderate", "selective"),
        ("dense", "selective"),
        ("unknown", "selective"),
        (None, "selective"),
    ],
)
def test_generated_engagement_context_uses_run_stable_density_default(
    density,
    expected,
) -> None:
    payload = _minimal_valid_v2_brief()
    if density is not None:
        payload["market_density"] = density

    context = normalize_generated_engagement_context(payload)

    assert context == {"selectivity_posture": expected}
    assert payload["engagement_context"] is context
    validate_v2_brief(payload)


def test_generated_engagement_context_preserves_valid_authored_posture() -> None:
    payload = _minimal_valid_v2_brief()
    payload["market_density"] = "dense"
    payload["engagement_context"] = {
        "selectivity_posture": "coverage",
        "talent_bar_statement": "Surface every plausible specialist.",
    }

    normalize_generated_engagement_context(payload)

    assert payload["engagement_context"] == {
        "selectivity_posture": "coverage",
        "talent_bar_statement": "Surface every plausible specialist.",
    }


def test_validate_v2_brief_accepts_minimal_engagement_context() -> None:
    payload = _minimal_valid_v2_brief()
    payload["engagement_context"] = {"selectivity_posture": "coverage"}

    validate_v2_brief(payload)


def test_validate_v2_brief_accepts_optional_engagement_strings() -> None:
    payload = _minimal_valid_v2_brief()
    payload["engagement_context"] = {
        "hiring_company": "",
        "engagement_description": "A focused product search.",
        "talent_bar_statement": "",
        "selectivity_posture": "selective",
    }

    validate_v2_brief(payload)


@pytest.mark.parametrize(
    "engagement_context",
    [
        None,
        [],
        {},
        {"selectivity_posture": "balanced"},
        {"selectivity_posture": "selective", "hiring_company": 42},
        {"selectivity_posture": "coverage", "engagement_description": None},
        {"selectivity_posture": "coverage", "talent_bar_statement": []},
    ],
)
def test_validate_v2_brief_rejects_malformed_engagement_context(
    engagement_context,
) -> None:
    payload = _minimal_valid_v2_brief()
    payload["engagement_context"] = engagement_context

    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)

    assert any(
        key.startswith("engagement_context")
        for key in excinfo.value.invalid_keys
    )


def test_validate_v2_brief_rejects_non_dict() -> None:
    with pytest.raises(BriefSchemaError, match="JSON object"):
        validate_v2_brief("not a dict")  # type: ignore[arg-type]


def test_validate_v2_brief_reports_missing_required_keys() -> None:
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief({"role_title": "Missing Required"})

    err = excinfo.value
    # Both required keys missing should appear in the structured error.
    assert "capability_areas" in err.missing_keys
    assert "depth_distinction" in err.missing_keys


def test_validate_v2_brief_rejects_empty_capability_areas() -> None:
    payload = _minimal_valid_v2_brief()
    payload["capability_areas"] = []
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "capability_areas" in excinfo.value.invalid_keys


def test_validate_v2_brief_rejects_capability_area_missing_description() -> None:
    payload = _minimal_valid_v2_brief()
    payload["capability_areas"][0].pop("description")
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    # The exact path varies; match on the prefix.
    assert any("capability_areas[0]" in k for k in excinfo.value.invalid_keys)


def test_validate_v2_brief_rejects_depth_distinction_missing_field() -> None:
    payload = _minimal_valid_v2_brief()
    payload["depth_distinction"].pop("edge_case_guidance")
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "depth_distinction.edge_case_guidance" in excinfo.value.invalid_keys


def test_validate_v2_brief_does_not_raise_on_unknown_keys() -> None:
    """Validation is structural ('is this V2-shaped'), not key-clean."""

    payload = _minimal_valid_v2_brief()
    payload["something_unknown"] = {"foo": "bar"}
    payload["must_haves"] = ["legacy"]  # deprecated key
    validate_v2_brief(payload)  # No raise.


# ---------------------------------------------------------------------------
# merge_legacy_brief
# ---------------------------------------------------------------------------


def test_merge_legacy_brief_splits_v2_legacy_unknown_buckets() -> None:
    payload = {
        # V2 recognized.
        "role_title": "Engineer",
        "capability_areas": [{"name": "x", "description": "y"}],
        "depth_distinction": {
            "builder_definition": "a",
            "user_definition": "b",
            "edge_case_guidance": "c",
        },
        # Deprecated.
        "must_haves": ["legacy thing"],
        "save_instructions": {"foo": "bar"},
        # Unknown (recruiter-authored, neither V2 nor known-deprecated).
        "recruiter_notes": "personal annotation",
    }

    merged = merge_legacy_brief(payload)

    assert isinstance(merged, MergedBrief)
    # V2 bucket has all V2-recognized keys.
    assert "role_title" in merged.v2_data
    assert "capability_areas" in merged.v2_data
    assert "depth_distinction" in merged.v2_data
    # Deprecated and unknown are NOT in v2_data.
    assert "must_haves" not in merged.v2_data
    assert "recruiter_notes" not in merged.v2_data
    # Deprecated bucket lists exactly the deprecated keys present.
    assert set(merged.deprecated_keys) == {"must_haves", "save_instructions"}
    # Unknown bucket lists the recruiter-authored key.
    assert merged.unknown_keys == ("recruiter_notes",)
    # Preserved legacy carries everything outside V2 — Fork B says
    # nothing gets dropped silently on edit.
    assert merged.preserved_legacy["must_haves"] == ["legacy thing"]
    assert merged.preserved_legacy["recruiter_notes"] == "personal annotation"


def test_merge_routes_experience_ceiling_fields_to_v2_bucket() -> None:
    payload = _minimal_valid_v2_brief()
    payload.update(
        {
            "maximum_years_experience": 10,
            "experience_measure": "total professional experience",
            "maximum_years_experience_is_hard": False,
        }
    )

    merged = merge_legacy_brief(payload)

    assert merged.v2_data["maximum_years_experience"] == 10
    assert merged.v2_data["experience_measure"] == "total professional experience"
    assert merged.v2_data["maximum_years_experience_is_hard"] is False
    assert not {
        "maximum_years_experience",
        "experience_measure",
        "maximum_years_experience_is_hard",
    }.intersection(merged.unknown_keys)


def test_merge_legacy_brief_preserves_legacy_for_writeback() -> None:
    """The preserved_legacy dict is what D2's PUT path uses to write back
    the full brief on disk — no recruiter-authored history dropped."""

    payload = {
        "capability_areas": [{"name": "x", "description": "y"}],
        "depth_distinction": {
            "builder_definition": "a",
            "user_definition": "b",
            "edge_case_guidance": "c",
        },
        "calibration_examples": [{"who": "A", "verdict": "save"}],
        "inference_save_rules": {"rule_a": "Save when X."},
    }

    merged = merge_legacy_brief(payload)

    # Both deprecated keys preserved verbatim.
    assert merged.preserved_legacy["calibration_examples"] == [
        {"who": "A", "verdict": "save"}
    ]
    assert merged.preserved_legacy["inference_save_rules"] == {
        "rule_a": "Save when X."
    }


def test_merge_legacy_brief_rejects_non_dict() -> None:
    with pytest.raises(BriefSchemaError, match="JSON object"):
        merge_legacy_brief(["not", "a", "dict"])  # type: ignore[arg-type]


def test_merge_legacy_brief_handles_empty_payload() -> None:
    merged = merge_legacy_brief({})
    assert merged.v2_data == {}
    assert merged.deprecated_keys == ()
    assert merged.unknown_keys == ()
    assert merged.preserved_legacy == {}


# ---------------------------------------------------------------------------
# deprecation_message
# ---------------------------------------------------------------------------


def test_deprecation_message_returns_rationale_for_known_key() -> None:
    msg = deprecation_message("must_haves")
    assert msg is not None
    assert "capability_areas" in msg  # Points at the V2 replacement.


def test_deprecation_message_returns_none_for_unknown_key() -> None:
    assert deprecation_message("totally_unknown_key") is None


def test_deprecation_manifest_has_at_least_one_version() -> None:
    """Sanity check: the manifest is non-empty so future deprecations
    have a place to land. If this fails, the module's premise is gone."""

    assert len(DEPRECATED_KEYS_BY_VERSION) >= 1
    for version, keys in DEPRECATED_KEYS_BY_VERSION.items():
        assert isinstance(version, str)
        assert isinstance(keys, dict)
        for key, rationale in keys.items():
            assert isinstance(key, str)
            assert isinstance(rationale, str)
            assert len(rationale) > 0  # No empty rationales — that's the whole point.


# ---------------------------------------------------------------------------
# Executive Search module (Slice 1)
# ---------------------------------------------------------------------------


def test_exec_search_keys_recognized_in_v2_schema() -> None:
    """The four top-level exec_search keys are V2-recognized."""

    for key in (
        "confidentiality_class",
        "prior_search",
        "board_signals",
        "executive_movement_window_days",
    ):
        assert key in RECOGNIZED_V2_KEYS, f"{key!r} missing from RECOGNIZED_V2_KEYS"


def test_exec_search_source_config_buckets_present() -> None:
    """Both source-config dicts carry an `exec_search` bucket (empty for Slice 1)."""

    assert "exec_search" in SOURCE_CONFIG_REQUIRED_KEYS_BY_SOURCE
    assert "exec_search" in SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE
    assert SOURCE_CONFIG_REQUIRED_KEYS_BY_SOURCE["exec_search"] == frozenset()
    assert SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE["exec_search"] == frozenset()


def test_recognized_confidentiality_classes_constant() -> None:
    assert RECOGNIZED_CONFIDENTIALITY_CLASSES == frozenset(
        {"open", "referenceable", "blind"}
    )


def test_validate_v2_brief_accepts_omitted_confidentiality_class() -> None:
    """Default behavior: brief without `confidentiality_class` validates fine."""

    payload = _minimal_valid_v2_brief()
    validate_v2_brief(payload)


@pytest.mark.parametrize(
    "value", ["open", "referenceable", "blind"]
)
def test_validate_v2_brief_accepts_recognized_confidentiality_class(value: str) -> None:
    payload = _minimal_valid_v2_brief()
    payload["confidentiality_class"] = value
    validate_v2_brief(payload)


def test_validate_v2_brief_rejects_unknown_confidentiality_class() -> None:
    payload = _minimal_valid_v2_brief()
    payload["confidentiality_class"] = "secret"
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "confidentiality_class" in excinfo.value.invalid_keys


def test_validate_v2_brief_rejects_non_string_confidentiality_class() -> None:
    payload = _minimal_valid_v2_brief()
    payload["confidentiality_class"] = 1
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "confidentiality_class" in excinfo.value.invalid_keys


def test_validate_v2_brief_treats_empty_confidentiality_class_as_default() -> None:
    """An empty string is treated as omitted (defaults to 'open' downstream)."""

    payload = _minimal_valid_v2_brief()
    payload["confidentiality_class"] = ""
    validate_v2_brief(payload)


def test_validate_v2_brief_accepts_exec_search_source_config_bucket() -> None:
    """Brief with `source_config.exec_search` validates (Slice 1: empty bucket)."""

    payload = _minimal_valid_v2_brief()
    payload["source_config"] = {"exec_search": {}}
    validate_v2_brief(payload)


def test_merge_legacy_brief_routes_exec_search_keys_into_v2() -> None:
    """The four exec_search top-level keys land in `v2_data`, not `preserved_legacy`."""

    payload = {
        "capability_areas": [{"name": "x", "description": "y"}],
        "depth_distinction": {
            "builder_definition": "a",
            "user_definition": "b",
            "edge_case_guidance": "c",
        },
        "confidentiality_class": "blind",
        "prior_search": {"ruled_out_urls": ["https://linkedin.com/in/foo"]},
        "board_signals": {"relevant_board_companies": ["Acme"]},
        "executive_movement_window_days": 90,
    }

    merged = merge_legacy_brief(payload)

    assert merged.v2_data["confidentiality_class"] == "blind"
    assert merged.v2_data["prior_search"]["ruled_out_urls"] == [
        "https://linkedin.com/in/foo"
    ]
    assert merged.v2_data["board_signals"]["relevant_board_companies"] == ["Acme"]
    assert merged.v2_data["executive_movement_window_days"] == 90
    assert merged.preserved_legacy == {}
