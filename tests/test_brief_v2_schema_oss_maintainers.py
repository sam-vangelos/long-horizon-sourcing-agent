"""OSS Maintainers module Slice 2 — V2 brief schema coverage.

Asserts that:

- ``validate_v2_brief`` accepts a brief with ``target_projects=[...]``,
  ``target_stacks=[...]``, and ``maintainership_level="maintainer"``.
- The validator rejects malformed shapes: non-list lists, non-string
  list items, unknown ``maintainership_level`` values.
- The recognized maintainership-level set matches the OSS Maintainers
  Module Spec §8 contract (``contributor`` / ``maintainer`` /
  ``project_lead``).
- ``source_config.github`` stays empty per spec §8 — these fields are
  evaluation inputs, not save-destination semantics.
"""

from __future__ import annotations

import pytest

from shared.brief_v2_schema import (
    MAINTAINERSHIP_LEVEL_ORDER,
    RECOGNIZED_MAINTAINERSHIP_LEVELS,
    SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE,
    BriefSchemaError,
    validate_v2_brief,
)


def _minimal_valid_v2_brief() -> dict:
    return {
        "role_title": "Staff infra engineer",
        "capability_areas": [
            {
                "name": "Container orchestration",
                "description": "Hands-on contributor to Kubernetes-tier projects.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "Has merge authority on a target project.",
            "user_definition": "Imports the library; doesn't ship to it.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }


def test_recognized_maintainership_levels_match_spec_contract() -> None:
    expected = frozenset({"contributor", "maintainer", "project_lead"})
    assert RECOGNIZED_MAINTAINERSHIP_LEVELS == expected


def test_maintainership_level_order_is_low_to_high() -> None:
    assert MAINTAINERSHIP_LEVEL_ORDER == (
        "contributor",
        "maintainer",
        "project_lead",
    )


def test_source_config_github_stays_empty_per_spec_section_8() -> None:
    """Spec §8: target_projects are evaluation inputs, not save-dest config."""

    assert SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE["github"] == frozenset()


def test_validate_accepts_brief_with_oss_maintainer_fields() -> None:
    brief = _minimal_valid_v2_brief()
    brief["target_projects"] = ["kubernetes/kubernetes", "etcd-io/etcd"]
    brief["target_stacks"] = ["go", "container-orchestration"]
    brief["maintainership_level"] = "maintainer"
    validate_v2_brief(brief)


def test_validate_accepts_brief_without_oss_maintainer_fields() -> None:
    """Behavior-preserving: classic briefs (no target_projects) still validate."""

    brief = _minimal_valid_v2_brief()
    validate_v2_brief(brief)


def test_validate_accepts_each_recognized_maintainership_level() -> None:
    for level in RECOGNIZED_MAINTAINERSHIP_LEVELS:
        brief = _minimal_valid_v2_brief()
        brief["maintainership_level"] = level
        validate_v2_brief(brief)


def test_validate_rejects_unknown_maintainership_level() -> None:
    brief = _minimal_valid_v2_brief()
    brief["maintainership_level"] = "core_team"  # not in the enum
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "maintainership_level" in exc_info.value.invalid_keys


def test_validate_rejects_non_list_target_projects() -> None:
    brief = _minimal_valid_v2_brief()
    brief["target_projects"] = "kubernetes/kubernetes"  # string, not list
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "target_projects" in exc_info.value.invalid_keys


def test_validate_rejects_target_projects_with_non_string_entries() -> None:
    brief = _minimal_valid_v2_brief()
    brief["target_projects"] = ["kubernetes/kubernetes", 42]  # mixed types
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "target_projects" in exc_info.value.invalid_keys


def test_validate_rejects_non_list_target_stacks() -> None:
    brief = _minimal_valid_v2_brief()
    brief["target_stacks"] = "go"  # string, not list
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "target_stacks" in exc_info.value.invalid_keys


def test_validate_rejects_target_stacks_with_non_string_entries() -> None:
    brief = _minimal_valid_v2_brief()
    brief["target_stacks"] = ["go", None]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "target_stacks" in exc_info.value.invalid_keys


def test_validate_accepts_empty_lists_and_empty_level() -> None:
    """Defaults / empty values pass validation (recruiter cleared them)."""

    brief = _minimal_valid_v2_brief()
    brief["target_projects"] = []
    brief["target_stacks"] = []
    brief["maintainership_level"] = ""
    validate_v2_brief(brief)


def test_validate_accepts_none_oss_fields() -> None:
    """Optional fields: explicit `None` (rare in JSON) is accepted as 'absent'."""

    brief = _minimal_valid_v2_brief()
    brief["target_projects"] = None
    brief["target_stacks"] = None
    brief["maintainership_level"] = None
    validate_v2_brief(brief)
