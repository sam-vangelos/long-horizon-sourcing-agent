"""Researcher module Slice 1 — V2 brief schema coverage.

Asserts that:

- `validate_v2_brief` accepts a brief with `target_modules=["researcher"]`
  and `source_config.researcher.research_topics=[...]` (and the other 5
  recognized keys with their natural shapes — list, str, int).
- The recognized key set matches the Researcher Module Spec Slice 1
  contract (`research_topics`, `conference_allowlist`, `discipline`,
  `h_index_floor`, `papers_in_window_floor`, `papers_in_window_months`).
- Per-source value-type semantics (e.g., research_topics being a list
  of strings) are NOT enforced by the structural validator — those
  live in the source-specific code path. The Slice 1 validator widening
  removed an over-narrow "must be string" check; this test pins the
  new looser contract.
"""

from __future__ import annotations

import pytest

from shared.brief_v2_schema import (
    SOURCE_CONFIG_KEY_VALUE_TYPES,
    SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE,
    BriefSchemaError,
    validate_v2_brief,
)


def _minimal_valid_v2_brief() -> dict:
    return {
        "role_title": "Frontier-lab researcher",
        "capability_areas": [
            {
                "name": "Post-training research",
                "description": "Publishes original work on RLHF / DPO / SFT.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "First-author publications at canonical venues.",
            "user_definition": "Cites papers but doesn't publish.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }


def test_recognized_keys_match_spec_slice_1_contract() -> None:
    expected = frozenset(
        {
            "research_topics",
            "conference_allowlist",
            "discipline",
            "h_index_floor",
            "papers_in_window_floor",
            "papers_in_window_months",
        }
    )
    assert SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE["researcher"] == expected


def test_validate_accepts_researcher_target_modules_with_full_source_config() -> None:
    """The full set of recognized researcher keys, in their natural shape."""

    payload = _minimal_valid_v2_brief()
    payload["target_modules"] = ["researcher"]
    payload["source_config"] = {
        "researcher": {
            "research_topics": ["RLHF", "DPO", "agent infrastructure"],
            "conference_allowlist": ["NeurIPS", "ICML", "ICLR"],
            "discipline": "nlp",
            "h_index_floor": 8,
            "papers_in_window_floor": 3,
            "papers_in_window_months": 24,
        }
    }
    validate_v2_brief(payload)  # No raise.


def test_validate_accepts_researcher_with_only_discipline() -> None:
    """Discipline alone is a complete configuration (Opinion 7: discipline
    is the load-bearing field; floors resolve from defaults).
    """

    payload = _minimal_valid_v2_brief()
    payload["target_modules"] = ["researcher"]
    payload["source_config"] = {"researcher": {"discipline": "nlp"}}
    validate_v2_brief(payload)  # No raise.


def test_validate_accepts_researcher_with_only_research_topics() -> None:
    """Research topics alone (no discipline, no floors) is also valid —
    universal minimum applies at evaluation time.
    """

    payload = _minimal_valid_v2_brief()
    payload["target_modules"] = ["researcher"]
    payload["source_config"] = {
        "researcher": {"research_topics": ["mechanistic interpretability"]}
    }
    validate_v2_brief(payload)  # No raise.


def test_validate_accepts_empty_researcher_source_config() -> None:
    """Empty researcher source_config is valid — universal minimum applies."""

    payload = _minimal_valid_v2_brief()
    payload["target_modules"] = ["researcher"]
    payload["source_config"] = {"researcher": {}}
    validate_v2_brief(payload)  # No raise.


def test_validate_accepts_researcher_alongside_linkedin() -> None:
    """A multi-module brief can carry both researcher + linkedin
    source_config simultaneously without conflict.
    """

    payload = _minimal_valid_v2_brief()
    payload["target_modules"] = ["linkedin", "researcher"]
    payload["source_config"] = {
        "linkedin": {"project_id": "12345"},
        "researcher": {
            "discipline": "ml_general",
            "research_topics": ["inference systems"],
        },
    }
    validate_v2_brief(payload)  # No raise.


def test_key_value_types_match_spec_slice_1_contract() -> None:
    """Pin the per-key typed contract added in Slice 1 alongside the
    recognized set. Per-source value-type validation lives in
    `SOURCE_CONFIG_KEY_VALUE_TYPES`; researcher's keys carry a mix of
    list / str / int per the natural recruiter input shape.
    """

    expected = {
        "research_topics": list,
        "conference_allowlist": list,
        "discipline": str,
        "h_index_floor": int,
        "papers_in_window_floor": int,
        "papers_in_window_months": int,
    }
    assert SOURCE_CONFIG_KEY_VALUE_TYPES["researcher"] == expected


def test_validate_rejects_research_topics_as_string() -> None:
    """research_topics is a list of strings — passing a single string is
    a structural error caught by the validator (not a deeper semantic
    check at strategy time).
    """

    payload = _minimal_valid_v2_brief()
    payload["source_config"] = {"researcher": {"research_topics": "RLHF"}}
    with pytest.raises(BriefSchemaError) as exc:
        validate_v2_brief(payload)
    assert "source_config.researcher.research_topics" in exc.value.invalid_keys


def test_validate_rejects_h_index_floor_as_string() -> None:
    """h_index_floor must be an int, not a string."""

    payload = _minimal_valid_v2_brief()
    payload["source_config"] = {"researcher": {"h_index_floor": "8"}}
    with pytest.raises(BriefSchemaError) as exc:
        validate_v2_brief(payload)
    assert "source_config.researcher.h_index_floor" in exc.value.invalid_keys


def test_validate_rejects_discipline_as_list() -> None:
    """discipline is a single-select string (one of the eight enum
    values per Opinion 7); a list of values is structurally invalid.
    """

    payload = _minimal_valid_v2_brief()
    payload["source_config"] = {"researcher": {"discipline": ["nlp", "vision"]}}
    with pytest.raises(BriefSchemaError) as exc:
        validate_v2_brief(payload)
    assert "source_config.researcher.discipline" in exc.value.invalid_keys
