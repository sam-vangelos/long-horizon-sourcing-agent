"""The attainability tier: who is likely to leave their seat, read off the record.

The judge scores fit and never asks whether the person would plausibly leave
their current job. This rule answers that question on both slates with no
model call, from the current employer and the current title, and its inputs
come from the brief's ``attainability`` block with fixed defaults when the
brief carries none.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.attainability import (
    DEFAULT_LONG_SHOT_EMPLOYERS,
    DEFAULT_LONG_SHOT_TITLE_PATTERNS,
    LONG_SHOT,
    REACHABLE,
    AttainabilityRules,
    classify_attainability,
)
from shared.brief_loader import load_brief
from shared.brief_v2_schema import (
    RECOGNIZED_V2_KEYS,
    BriefSchemaError,
    validate_v2_brief,
)
from tests.test_exa_orchestrator import _brief_dict


DEFAULTS = AttainabilityRules.from_brief({})


@pytest.mark.parametrize(
    "company, title, headline, expected",
    [
        ("OpenAI", "Member of Technical Staff", "", (LONG_SHOT, "current employer: OpenAI")),
        # A past employer named in the headline is an alumnus, not a current seat.
        ("Acme Robotics", "ML Engineer", "ML Engineer | ex-OpenAI", (REACHABLE, "")),
        ("", "ML Engineer | ex-OpenAI", "", (REACHABLE, "")),
        ("Stealth", "Co-Founder", "", (LONG_SHOT, "title: Co-Founder")),
        ("Acme", "Cofounder & CTO", "", (LONG_SHOT, "title: Cofounder")),
        # "founding" is not "founder": a founding hire is exactly who this search wants.
        ("Acme", "Founding Account Manager", "", (REACHABLE, "")),
        ("Meta Reality Labs", "Product Manager", "", (LONG_SHOT, "current employer: Meta")),
        ("Metaview", "Product Manager", "", (REACHABLE, "")),
        ("Salesforce", "Senior Product Manager", "", (REACHABLE, "")),
        ("Acme", "Chief Technology Officer", "", (LONG_SHOT, "title: Chief Technology")),
        ("Acme", "CEO", "", (LONG_SHOT, "title: CEO")),
        ("", "Research Engineer", "Research Engineer @ Anthropic", (LONG_SHOT, "current employer: Anthropic")),
        ("", "", "", (REACHABLE, "")),
    ],
)
def test_classify(company, title, headline, expected) -> None:
    assert classify_attainability(company, title, DEFAULTS, headline=headline) == expected


# A founder-level title counts only when it IS the seat: at the start of the
# title or of a segment, and not possessive. A live client slate sorted the
# people around a founder (chief of staff, the CEO's office) as long shots.
@pytest.mark.parametrize(
    "title, reason",
    [
        ("Co-Founder & CTO", "Co-Founder"),
        ("CEO", "CEO"),
        ("Founder", "Founder"),
        ("Co-founder, Head of Sales", "Co-founder"),
        ("Chief Executive Officer", "Chief Executive"),
        ("Founder and CEO", "Founder"),
        ("CTO / Co-Founder", "Co-Founder"),
        ("Head of Sales | CEO", "CEO"),
        ("Advisor · Founder", "Founder"),
        ("Head of Product - Co-Founder", "Co-Founder"),
        ("Head of Sales (Co-Founder)", "Co-Founder"),
        # A qualifier that leaves the seat a founder's seat.
        ("Technical Co-Founder", "Technical Co-Founder"),
        ("Interim CEO", "Interim CEO"),
    ],
)
def test_founder_level_title_is_a_long_shot(title, reason) -> None:
    assert classify_attainability("Acme", title, DEFAULTS) == (LONG_SHOT, f"title: {reason}")


@pytest.mark.parametrize(
    "title",
    [
        "Chief of Staff to the CEO",
        "Office of the CTO",
        "Founder's Associate",
        "Founder’s Associate",
        "Assistant to the Founder",
        "CEO's Office",
        "Founding Account Manager",
        "Founding Engineer",
        "Executive Assistant to the Co-Founder",
        "Strategy & Operations, Office of the CEO",
        # A customer-facing seat that borrows the title.
        "Field CTO",
    ],
)
def test_title_around_a_founder_is_reachable(title) -> None:
    assert classify_attainability("Acme", title, DEFAULTS) == (REACHABLE, "")


def test_a_brief_pattern_still_matches_anywhere_in_the_title() -> None:
    """Segment-start is how the defaults are written, not a rule on every pattern."""

    rules = AttainabilityRules.from_brief(
        {"attainability": {"long_shot_title_patterns": ["\\bvp\\b"]}}
    )
    assert classify_attainability("Acme", "Senior VP, Sales", rules) == (LONG_SHOT, "title: VP")


def test_employer_match_is_reported_before_a_title_match() -> None:
    assert classify_attainability("xAI", "Co-Founder", DEFAULTS) == (
        LONG_SHOT,
        "current employer: xAI",
    )


def test_a_recorded_company_outranks_the_headline() -> None:
    """The live card that read as Meta staff: Meta is a past seat.

    The headline mixes present and past ("Ex-SWE @ Meta"), so it is read
    only when the record names no current company.
    """

    assert classify_attainability(
        "Mercor",
        "Product Lead",
        DEFAULTS,
        headline="Product Lead @ Mercor Applied AI || Ex-SWE @ Meta",
    ) == (REACHABLE, "")


def test_title_patterns_read_the_title_not_the_headline() -> None:
    """A headline is self-description; "founder mindset" is not a seat."""

    assert classify_attainability(
        "Acme", "Account Manager", DEFAULTS, headline="Founder at heart | GTM"
    ) == (REACHABLE, "")


def test_defaults_are_the_ruled_values() -> None:
    assert DEFAULT_LONG_SHOT_EMPLOYERS == (
        "OpenAI",
        "Anthropic",
        "Google DeepMind",
        "DeepMind",
        "xAI",
        "Meta",
    )
    qualifier = r"(?:(?:technical|stealth|solo|serial|interim|acting|founding)\s+)?"
    not_possessive = r"\b(?!['’]s\b|\s+(?:office|associate|assistant)\b)"
    assert DEFAULT_LONG_SHOT_TITLE_PATTERNS == tuple(
        "^" + qualifier + role + not_possessive
        for role in (
            "co-?founder",
            "founder",
            "ceo",
            "cto",
            "chief executive",
            "chief technology",
        )
    )
    assert DEFAULTS.long_shot_employers == DEFAULT_LONG_SHOT_EMPLOYERS
    assert DEFAULTS.long_shot_title_patterns == DEFAULT_LONG_SHOT_TITLE_PATTERNS


# --- the brief field ----------------------------------------------------------


def test_brief_rules_replace_the_defaults_key_by_key() -> None:
    rules = AttainabilityRules.from_brief(
        {"attainability": {"long_shot_employers": ["Vendor Co"]}}
    )

    assert rules.long_shot_employers == ("Vendor Co",)
    # The key the brief left out keeps its default.
    assert rules.long_shot_title_patterns == DEFAULT_LONG_SHOT_TITLE_PATTERNS
    assert classify_attainability("OpenAI", "Engineer", rules) == (REACHABLE, "")
    assert classify_attainability("Vendor Co", "Engineer", rules) == (
        LONG_SHOT,
        "current employer: Vendor Co",
    )


def test_an_empty_list_switches_that_half_of_the_rule_off() -> None:
    rules = AttainabilityRules.from_brief(
        {"attainability": {"long_shot_employers": [], "long_shot_title_patterns": []}}
    )

    assert classify_attainability("OpenAI", "Co-Founder", rules) == (REACHABLE, "")


@pytest.mark.parametrize(
    "block",
    [
        ["OpenAI"],
        "OpenAI",
        {"long_shot_employers": "OpenAI"},
        {"long_shot_employers": ["OpenAI", 3]},
        {"long_shot_employers": ["  "]},
        {"long_shot_title_patterns": ["(unclosed"]},
        {"long_shot_employer": ["OpenAI"]},
    ],
)
def test_malformed_field_is_rejected(block) -> None:
    brief = _brief_dict(attainability=block)

    with pytest.raises(BriefSchemaError) as caught:
        validate_v2_brief(brief)
    assert "attainability" in caught.value.invalid_keys
    with pytest.raises(ValueError):
        AttainabilityRules.from_brief(brief)


def test_malformed_field_fails_the_load(tmp_path: Path) -> None:
    """A typo must not silently fall back to the defaults at launch."""

    path = tmp_path / "brief.json"
    path.write_text(
        json.dumps(_brief_dict(attainability={"long_shot_employers": "OpenAI"}))
    )

    with pytest.raises(BriefSchemaError):
        load_brief(path)


def test_field_is_recognised_and_carried_by_the_loader(tmp_path: Path) -> None:
    assert "attainability" in RECOGNIZED_V2_KEYS
    block = {"long_shot_employers": ["Vendor Co"], "long_shot_title_patterns": ["\\bvp\\b"]}
    validate_v2_brief(_brief_dict(attainability=block))
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_brief_dict(attainability=block)))

    brief = load_brief(path)

    assert brief._new_brief.attainability == block
    rules = AttainabilityRules.from_brief(brief)
    assert rules.long_shot_employers == ("Vendor Co",)
    assert rules.long_shot_title_patterns == ("\\bvp\\b",)


def test_absent_field_loads_to_the_defaults(tmp_path: Path) -> None:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_brief_dict()))

    brief = load_brief(path)

    assert brief._new_brief.attainability == {}
    assert AttainabilityRules.from_brief(brief) == DEFAULTS
    assert AttainabilityRules.from_brief(brief._new_brief) == DEFAULTS


def test_a_loaded_v1_seed_supplies_its_rules(tmp_path: Path) -> None:
    """The LinkedIn seed is a V1 brief; its block is read off ``raw``."""

    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps(
            {
                "name": "seed",
                "role_title": "Founding Head of Data",
                "jd": "# Founding Head of Data",
                "attainability": {"long_shot_employers": ["Vendor Co"]},
            }
        )
    )

    brief = load_brief(path)

    assert brief.has_v2_schema is False
    assert AttainabilityRules.from_brief(brief).long_shot_employers == ("Vendor Co",)
