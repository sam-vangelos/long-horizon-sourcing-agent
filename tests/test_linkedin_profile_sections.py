"""Tests for linkedin.profile_sections — section anchor mapping."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock

from linkedin.profile_sections import (
    SECTION_SELECTORS,
    SectionAnchor,
    _map_sections,
    locate_sections,
)

_STABLE_SELECTOR_TOKENS = frozenset(
    {
        "summary-card",
        "experience-card",
        "position-item",
        "education-entity",
        "h1",
        "h2",
        "h3",
    }
)
_HASH_LIKE_TOKEN_RE = re.compile(r"[a-zA-Z]{20,}")


def test_locates_sections_in_document_order_from_captured_dom():
    raw_headings = [
        {"text": "Summary", "offset": 100},
        {"text": "Experience", "offset": 400},
        {"text": "Education", "offset": 1200},
        {"text": "Skills (3)", "offset": 1600},
    ]

    result = _map_sections(raw_headings)

    assert [anchor.name for anchor in result] == [
        "about",
        "experience",
        "education",
        "skills",
    ]
    assert all(isinstance(anchor, SectionAnchor) for anchor in result)
    offsets = [anchor.offset for anchor in result]
    assert offsets == [100, 400, 1200, 1600]
    assert offsets == sorted(offsets)


def test_missing_section_is_omitted_not_faked():
    raw_headings = [
        {"text": "Summary", "offset": 100},
        {"text": "Experience", "offset": 400},
        {"text": "Skills (2)", "offset": 900},
    ]

    result = _map_sections(raw_headings)

    assert [anchor.name for anchor in result] == ["about", "experience", "skills"]
    assert not any(anchor.name == "education" for anchor in result)


def test_returns_empty_list_on_an_unrecognized_container():
    empty_container = AsyncMock()
    empty_container.evaluate = AsyncMock(return_value=[])

    assert asyncio.run(locate_sections(empty_container)) == []

    raising_container = AsyncMock()
    raising_container.evaluate = AsyncMock(side_effect=RuntimeError("no panel"))

    assert asyncio.run(locate_sections(raising_container)) == []


def test_section_selectors_are_stable_not_hashed():
    for section_name, spec in SECTION_SELECTORS.items():
        for key, value in spec.items():
            if key == "headings":
                continue
            if value is None:
                continue
            selector = str(value)
            assert not _HASH_LIKE_TOKEN_RE.search(selector), (
                f"{section_name}.{key} contains hash-like token: {selector!r}"
            )
            for token in re.findall(r"[a-z][a-z0-9-]*", selector.lower()):
                if token in {"li", "div"}:
                    continue
                assert token in _STABLE_SELECTOR_TOKENS or "-" in token, (
                    f"{section_name}.{key} has unexpected selector token: {token!r}"
                )
