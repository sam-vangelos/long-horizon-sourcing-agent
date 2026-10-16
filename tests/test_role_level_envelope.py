"""A brief's ``role_level_envelope`` reaches the judge.

``shared/judgment/templates.py:_level_envelope_or_default`` has always read
``brief.role_level_envelope`` and rendered the brief's own target, above and
below lines when it was set. The loader never put the field on the brief, so
every brief got the default envelope, whose "Clearly above it: ... managing
managers, owning a function or portfolio" turns into a hard over-level rule
once ``maximum_years_experience_is_hard`` is true. The SPL competitors brief
needs its own envelope, because its role level spans second-line managers who
still deliver (review of fix/spl-exa-r3, 2026-10-10).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.brief_loader import load_brief
from shared.brief_v2_schema import (
    RECOGNIZED_V2_KEYS,
    BriefSchemaError,
    validate_v2_brief,
)
from shared.judgment.templates import (
    assemble_full_evaluation_system,
    assemble_full_evaluation_tool_system,
)

_DEFAULT_ABOVE = (
    "Clearly above it: people whose demonstrated current level is managing "
    "managers, owning a function or portfolio, or executive scope."
)

_ENVELOPE = {
    "target_level_statement": (
        "Individual operators through second-line managers who still own delivery."
    ),
    "over_level_looks_like": (
        "Founders, CEOs, VPs, and directors over an organization of managers."
    ),
    "under_level_looks_like": "Task contributors with no lead or review role.",
}


def _brief_dict(**overrides) -> dict:
    raw = {
        "role_title": "Strategic Project Lead",
        "role_level": "SPL / Senior SPL",
        "role_summary": "Owns delivery of coding data programs.",
        "geography": "United States",
        "linkedin_project": "Test Project",
        "capability_areas": [
            {"name": "Program delivery", "description": "Owned a program."}
        ],
        "depth_distinction": {
            "builder_definition": "Owned delivery.",
            "user_definition": "Attended delivery.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "non_fit_patterns": [],
        "employer_signal_rules": [],
        "engagement_context": {"selectivity_posture": "coverage"},
        "minimum_years_experience": 2,
        "maximum_years_experience": 18,
        "maximum_years_experience_is_hard": True,
        "experience_measure": "total years of professional work",
    }
    raw.update(overrides)
    return raw


def _load(tmp_path: Path, **overrides):
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_brief_dict(**overrides)))
    return load_brief(str(path))


@pytest.mark.parametrize(
    "assemble", [assemble_full_evaluation_system, assemble_full_evaluation_tool_system]
)
def test_the_brief_envelope_replaces_the_default_in_both_prompts(
    tmp_path: Path, assemble
) -> None:
    system = assemble(_load(tmp_path, role_level_envelope=_ENVELOPE)._new_brief)

    assert f"ROLE LEVEL: {_ENVELOPE['target_level_statement']}" in system
    assert f"ABOVE this role looks like: {_ENVELOPE['over_level_looks_like']}" in system
    assert f"BELOW this role looks like: {_ENVELOPE['under_level_looks_like']}" in system
    assert _DEFAULT_ABOVE not in system


def test_without_an_envelope_the_default_still_renders(tmp_path: Path) -> None:
    system = assemble_full_evaluation_system(_load(tmp_path)._new_brief)

    assert "ROLE LEVEL: SPL / Senior SPL." in system
    assert _DEFAULT_ABOVE in system


def test_the_envelope_is_a_recognized_v2_key() -> None:
    assert "role_level_envelope" in RECOGNIZED_V2_KEYS


@pytest.mark.parametrize(
    "envelope",
    [
        "Second-line managers are aligned.",
        [{"target_level_statement": "Operators."}],
        {"over_level_looks_like": "VPs."},
        {"target_level_statement": "  "},
        {"target_level_statement": "Operators.", "above": "VPs."},
        {"target_level_statement": ["Operators."]},
    ],
)
def test_a_malformed_envelope_fails_closed(tmp_path: Path, envelope) -> None:
    """A malformed envelope must not silently fall back to the default.

    The launch path never runs ``validate_v2_brief``, so the loader refuses
    it too: an envelope the judge never sees, while the brief says it is
    set, is the defect this file exists for.
    """

    with pytest.raises(BriefSchemaError):
        validate_v2_brief(_brief_dict(role_level_envelope=envelope))
    with pytest.raises(BriefSchemaError):
        _load(tmp_path, role_level_envelope=envelope)


@pytest.mark.parametrize("envelope", [None, {}])
def test_an_absent_envelope_is_valid(tmp_path: Path, envelope) -> None:
    validate_v2_brief(_brief_dict(role_level_envelope=envelope))
    assert _DEFAULT_ABOVE in assemble_full_evaluation_system(
        _load(tmp_path, role_level_envelope=envelope)._new_brief
    )
