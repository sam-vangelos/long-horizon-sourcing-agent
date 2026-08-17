"""Tests for ``shared.intake_conversation.sufficiency`` (Phase C4).

The sufficiency detector decides whether the brief is ready to file. The
C5 endpoint emits the wire event from this deterministic check (not from
LLM JSON), so the threshold has to be unambiguous and the missing list
must be actionable for the orchestrator's next-turn prompt.

Threshold under test (from the C4 phase block):
- role_title non-empty (not placeholder) AND
- ≥1 capability_areas entry with non-placeholder description AND
- (role_summary non-empty OR ≥1 depth_distinction sub-field populated)
"""

from __future__ import annotations

import pytest

from shared.intake_conversation.sufficiency import is_ready_to_compose


# -------------------------------------------------------------------------
# Not-ready cases
# -------------------------------------------------------------------------


def test_empty_draft_not_ready_with_full_missing_list() -> None:
    ready, missing = is_ready_to_compose({})
    assert ready is False
    assert "role_title" in missing
    assert any("capability_areas" in m for m in missing)
    assert any("role_summary" in m or "depth_distinction" in m for m in missing)


def test_non_dict_input_not_ready() -> None:
    ready, missing = is_ready_to_compose(None)  # type: ignore[arg-type]
    assert ready is False
    assert missing


def test_role_title_only_not_ready() -> None:
    ready, missing = is_ready_to_compose({"role_title": "Senior Tax Associate"})
    assert ready is False
    assert "role_title" not in missing
    assert any("capability_areas" in m for m in missing)


def test_role_title_plus_capability_without_summary_or_depth_not_ready() -> None:
    draft = {
        "role_title": "Senior Tax Associate",
        "capability_areas": [
            {"name": "Sales tax compliance", "description": "Owns SUT filings."}
        ],
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is False
    assert any("role_summary" in m or "depth_distinction" in m for m in missing)


def test_capability_with_placeholder_description_not_ready() -> None:
    """Placeholder descriptions don't count as 'real' for sufficiency."""

    draft = {
        "role_title": "Senior Tax Associate",
        "role_summary": "Owns multi-state tax filings.",
        "capability_areas": [
            {"name": "Sales tax", "description": "Core role scope"}  # placeholder
        ],
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is False
    assert any("capability_areas" in m for m in missing)


def test_capability_without_name_not_counted() -> None:
    """An item with description but no name doesn't satisfy sufficiency —
    sourcing needs a label too.
    """

    draft = {
        "role_title": "Senior Tax Associate",
        "role_summary": "Owns multi-state tax filings.",
        "capability_areas": [
            {"description": "Some real description but no name."}
        ],
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is False


def test_placeholder_role_title_not_ready() -> None:
    """role_title that's actually a JD-prose dump fails the placeholder gate."""

    long_jd = (
        "We are seeking a Senior Tax Associate to own our multi-state sales "
        "tax compliance function. The ideal candidate brings 5+ years of "
        "experience and a CPA. This is a hands-on role partnering with finance."
    )
    draft = {
        "role_title": long_jd,
        "role_summary": "Owns SUT filings.",
        "capability_areas": [
            {"name": "Sales tax", "description": "Owns SUT filings end-to-end."}
        ],
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is False
    assert "role_title" in missing


def test_whitespace_only_role_title_not_ready() -> None:
    draft = {
        "role_title": "   ",
        "role_summary": "Owns SUT filings.",
        "capability_areas": [
            {"name": "Sales tax", "description": "Owns SUT filings."}
        ],
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is False
    assert "role_title" in missing


# -------------------------------------------------------------------------
# Ready cases
# -------------------------------------------------------------------------


def test_role_title_plus_capability_plus_role_summary_ready() -> None:
    draft = {
        "role_title": "Senior Tax Associate",
        "role_summary": "Owns multi-state sales tax filings end-to-end.",
        "capability_areas": [
            {"name": "Sales tax compliance", "description": "Owns SUT filings."}
        ],
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is True
    assert missing == []


def test_role_title_plus_capability_plus_depth_distinction_ready() -> None:
    """role_summary OR depth_distinction satisfies the third clause."""

    draft = {
        "role_title": "Senior Tax Associate",
        "capability_areas": [
            {"name": "Sales tax", "description": "Owns SUT filings."}
        ],
        "depth_distinction": {
            "builder_definition": "Builds the close from scratch."
        },
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is True


def test_multiple_capability_areas_one_real_is_enough() -> None:
    """Threshold is ≥1 real capability_area; extras don't matter."""

    draft = {
        "role_title": "Senior Tax Associate",
        "role_summary": "Owns SUT filings.",
        "capability_areas": [
            {"name": "Sales tax", "description": "Owns SUT filings."},
            {"name": "Reporting"},  # no description — doesn't count
            {"name": "Other", "description": "Core role scope"},  # placeholder
        ],
    }
    ready, missing = is_ready_to_compose(draft)
    assert ready is True


def test_any_depth_distinction_subkey_satisfies() -> None:
    """Each of the three depth_distinction sub-fields independently
    satisfies the third clause.
    """

    base = {
        "role_title": "Senior Tax Associate",
        "capability_areas": [
            {"name": "Sales tax", "description": "Owns SUT filings."}
        ],
    }
    for subkey in ("builder_definition", "user_definition", "edge_case_guidance"):
        draft = dict(base, depth_distinction={subkey: "non-empty"})
        ready, _ = is_ready_to_compose(draft)
        assert ready, f"{subkey} should satisfy the third sufficiency clause"


# -------------------------------------------------------------------------
# Missing-list ordering
# -------------------------------------------------------------------------


def test_missing_list_orders_role_title_first() -> None:
    """When multiple things are missing, role_title comes first so the
    orchestrator can ask the most foundational question first.
    """

    ready, missing = is_ready_to_compose({})
    assert missing[0] == "role_title"
