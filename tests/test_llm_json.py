"""Tests for shared/llm_json.extract_json (CLO-437).

The helper pulls one JSON object or array out of a model reply that may wrap
it in a markdown fence or in prose, and it never repairs a truncated reply:
a cut-off document must raise, not come back as a silently closed fragment.
"""

from __future__ import annotations

import json

import pytest

from shared.llm_json import extract_json


BRIEF = {
    "role_title": "Senior Engineer",
    "capability_areas": [{"name": "systems", "signals": ["a", "b"]}],
    "non_fit_patterns": ["pattern {x}"],
}


def test_bare_json_object():
    assert extract_json(json.dumps(BRIEF)) == BRIEF


def test_fenced_at_start():
    raw = "```json\n" + json.dumps(BRIEF, indent=2) + "\n```"

    assert extract_json(raw) == BRIEF


def test_prose_then_fence_then_prose():
    raw = (
        "Here is the brief:\n\n```json\n"
        + json.dumps(BRIEF, indent=2)
        + "\n```\n\nLet me know if you need changes."
    )

    assert extract_json(raw) == BRIEF


def test_prose_then_bare_object_then_trailing_prose_with_braces():
    raw = (
        "Sure [draft v2] - the brief follows.\n"
        + json.dumps(BRIEF, indent=2)
        + "\nNote: I left {placeholders} where the JD was silent."
    )

    assert extract_json(raw) == BRIEF


def test_array():
    raw = "The list:\n[1, 2, {\"a\": 3}]\nDone."

    assert extract_json(raw) == [1, 2, {"a": 3}]


@pytest.mark.parametrize(
    "raw",
    [
        # Cut off mid-document: the inner objects are complete, the outer is not.
        json.dumps(BRIEF, indent=2)[:-40],
        # Cut off inside a string value.
        '{"role_title": "Senior Eng',
        # Fenced, cut off before the closing fence.
        "```json\n" + json.dumps(BRIEF, indent=2)[:-25],
        # Prose, then a truncated document.
        "Here is the brief:\n" + json.dumps(BRIEF)[:-3],
    ],
    ids=["mid-document", "mid-string", "unclosed-fence", "prose-then-truncated"],
)
def test_truncated_json_raises_and_is_never_repaired(raw):
    with pytest.raises(json.JSONDecodeError):
        extract_json(raw)


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not json at all", "I could not produce a brief.", "42", '"text"'],
)
def test_no_json_raises_json_decode_error(raw):
    with pytest.raises(json.JSONDecodeError):
        extract_json(raw)
