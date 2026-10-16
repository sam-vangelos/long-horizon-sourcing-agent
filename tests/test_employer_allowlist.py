"""A brief's ``employer_allowlist`` bounds a campaign to named current employers.

A competitor-sourcing campaign sources only people
CURRENTLY employed at the hiring company's competitors in AI training data and evaluation.
LinkedIn Recruiter's company facet matches current OR past employees, and
three code paths can run a string with no competitor facet at all, so the
restriction cannot rest on the strategist. The allowlist is the deterministic
half: a seed field, inert for every brief that lacks it, read by the facial
gate, the post-profile gate, and the per-string facet stamp.

The matcher is deliberately generous in the in-scope direction. A false
"out of scope" settles a person as FACIAL_NO for good; a false "in scope" costs
one judge call. So a truncated company ("Invisible"), a punctuation variant
("Pareto.AI"), a contractor brand ("Outlier"), and an employment-type suffix
("AfterQuery Experts · Contract") all read as in scope, and a current employer
the card does not name, or names only as "Self-employed", passes to the judge.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SEED_ALLOWLIST = {
    "facet_groups": [
        ["Mercor"],
        ["Handshake"],
        ["Surge AI", "AfterQuery"],
        ["Scale AI"],
        ["Invisible Technologies", "Pareto AI", "Snorkel AI", "Labelbox"],
        ["Appen", "TELUS Digital", "Uber AI Solutions", "CloudFactory", "Defined.ai"],
    ],
    "aliases": {
        "Outlier": "Scale AI",
        "Remotasks": "Scale AI",
        "DataAnnotation": "Surge AI",
        "DataAnnotation.tech": "Surge AI",
        "Alignerr": "Labelbox",
        "AfterQuery Experts": "AfterQuery",
        "Handshake AI": "Handshake",
        "TELUS International AI": "TELUS Digital",
        "Mercor Intelligence": "Mercor",
    },
    "_why": "operator note, ignored by the parser",
}


def _allowlist():
    from shared.employer_allowlist import parse_employer_allowlist

    allowlist = parse_employer_allowlist(SEED_ALLOWLIST)
    assert allowlist is not None
    return allowlist


# ---------------------------------------------------------------------------
# Parsing: a well-formed field or nothing
# ---------------------------------------------------------------------------


def test_the_seed_field_parses_into_names_groups_and_aliases() -> None:
    allowlist = _allowlist()

    assert allowlist.names[:4] == ("Mercor", "Handshake", "Surge AI", "AfterQuery")
    assert len(allowlist.facet_groups) == 6
    assert allowlist.aliases["Outlier"] == "Scale AI"


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "Mercor",
        {},
        {"facet_groups": []},
        {"facet_groups": "Mercor"},
        {"facet_groups": [[]]},
        {"facet_groups": [["Mercor", 3]]},
        {"facet_groups": [["Mercor"]], "aliases": ["Outlier"]},
        # An alias must name a listed employer, or it can never put anyone in scope.
        {"facet_groups": [["Mercor"]], "aliases": {"Outlier": "Scale AI"}},
        # A group larger than the per-facet cap cannot ride one string.
        {"facet_groups": [[f"Vendor {i}" for i in range(13)]]},
    ],
)
def test_a_malformed_field_parses_to_nothing(value) -> None:
    from shared.employer_allowlist import parse_employer_allowlist

    assert parse_employer_allowlist(value) is None


def test_a_brief_without_the_field_is_inert() -> None:
    from shared.employer_allowlist import employer_allowlist_from_brief

    assert employer_allowlist_from_brief(SimpleNamespace(raw={})) is None
    assert employer_allowlist_from_brief(SimpleNamespace(raw=MagicMock())) is None
    assert employer_allowlist_from_brief(SimpleNamespace()) is None
    assert employer_allowlist_from_brief(
        SimpleNamespace(raw={"employer_allowlist": SEED_ALLOWLIST})
    ) is not None


# ---------------------------------------------------------------------------
# canonical_for: which listed employer a company or facet value names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "company, canonical",
    [
        ("Mercor", "Mercor"),
        ("mercor", "Mercor"),
        ("Mercor Intelligence", "Mercor"),
        ("Handshake AI", "Handshake"),
        ("Scale AI", "Scale AI"),
        ("Outlier", "Scale AI"),
        ("Outlier AI", "Scale AI"),
        ("Remotasks", "Scale AI"),
        ("DataAnnotation", "Surge AI"),
        ("DataAnnotation.tech", "Surge AI"),
        ("Data Annotation", "Surge AI"),
        ("Alignerr", "Labelbox"),
        ("AfterQuery Experts · Contract", "AfterQuery"),
        ("Pareto.AI", "Pareto AI"),
        ("Invisible", "Invisible Technologies"),
        ("Invisible Technologies Inc.", "Invisible Technologies"),
        ("Surge", "Surge AI"),
        ("Snorkel", "Snorkel AI"),
        ("Cloud Factory", "CloudFactory"),
        ("Defined.ai", "Defined.ai"),
        ("TELUS International AI", "TELUS Digital"),
        ("Uber AI Solutions", "Uber AI Solutions"),
    ],
)
def test_a_listed_employer_is_recognized_in_its_common_spellings(company, canonical) -> None:
    assert _allowlist().canonical_for(company) == canonical


@pytest.mark.parametrize(
    "company",
    [
        "Google",
        "xAI",
        "Meta",
        # A current Uber employee counts only in an Uber AI Solutions seat.
        "Uber",
        "Amazon Web Services (AWS)",
        "",
        None,
    ],
)
def test_an_unlisted_employer_is_not_recognized(company) -> None:
    assert _allowlist().canonical_for(company) is None


# ---------------------------------------------------------------------------
# The scope verdict over a person's CURRENT employer signals
# ---------------------------------------------------------------------------


def test_a_listed_current_company_is_in_scope() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Mercor Intelligence"], headlines=["AI Ops"], allowlist=_allowlist()
    )

    assert verdict.in_scope is True
    assert verdict.employer == "Mercor"


def test_an_unlisted_current_company_is_out_of_scope_and_named() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Google"], headlines=["Program Manager at Google"], allowlist=_allowlist()
    )

    assert verdict.in_scope is False
    assert (verdict.field, verdict.value) == ("current_company", "Google")


def test_a_second_current_seat_at_a_competitor_keeps_the_person_in_scope() -> None:
    """A Mercor contract held alongside a day job: any current competitor seat counts."""
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Stanford University", "Outlier"], headlines=[], allowlist=_allowlist()
    )

    assert verdict.in_scope is True
    assert verdict.employer == "Scale AI"


def test_a_headline_naming_a_competitor_as_employer_keeps_the_person_in_scope() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Acme Staffing"],
        headlines=["Engagement Manager at Mercor | ex-Google"],
        allowlist=_allowlist(),
    )

    assert verdict.in_scope is True


def test_a_past_competitor_in_the_headline_does_not_put_the_person_in_scope() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Google"],
        headlines=["Program Manager at Google | ex-Strategic Projects Lead at Scale AI"],
        allowlist=_allowlist(),
    )

    assert verdict.in_scope is False


@pytest.mark.parametrize(
    "companies",
    [
        [], [""], ["Self-employed"], ["Freelance"], ["Stealth Startup"], ["Confidential"],
        # LinkedIn's placeholder companies for an unnamed employer.
        ["Startup"], ["Stealth"], ["Stealth Startup · Full-time"], ["Startup · Full-time"],
        ["Early-stage startup"], ["Stealth Mode"],
    ],
)
def test_an_unidentified_current_employer_passes_to_the_judge(companies) -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(companies=companies, headlines=[], allowlist=_allowlist())

    assert verdict.in_scope is None


@pytest.mark.parametrize("company", ["Startup Grind", "Startupbootcamp", "Stealth Health"])
def test_a_named_company_that_contains_a_placeholder_word_is_identified(company) -> None:
    """Only the whole placeholder is unidentified; a real company is not."""
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(companies=[company], headlines=[], allowlist=_allowlist())

    assert verdict.in_scope is False


# ---------------------------------------------------------------------------
# Card and profile signals: what LinkedIn says the person does NOW
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry, company",
    [
        ("Senior Strategic Projects Lead at Mercor (2026 – Present)", "Mercor"),
        ("Human Data Strategy at Encord · 2025 – Present", "Encord"),
        ("AI Trainer at Outlier (2024 – present)", "Outlier"),
        ("Program Manager at Google (2021 – 2024)", None),
        ("Team Lead at Scale AI (2023 – 2025)", None),
        ("Freeform text with no employer", None),
    ],
)
def test_a_card_entry_names_its_company_only_when_it_is_current(entry, company) -> None:
    from shared.employer_allowlist import present_entry_company

    assert present_entry_company(entry) == company


@pytest.mark.parametrize(
    "entity, listed",
    [
        ("Mercor", "Mercor"),
        ("Scale AI", "Scale AI"),
        ("Invisible Technologies", "Invisible Technologies"),
        ("Invisible Technologies Inc.", "Invisible Technologies"),
        ("Pareto.AI", "Pareto AI"),
        ("Defined.ai", "Defined.ai"),
        ("Handshake AI", "Handshake"),
        ("Outlier", "Scale AI"),
        ("Handshake Partners", None),
        ("Uber", None),
        ("Mercor Health", None),
        ("", None),
    ],
)
def test_a_landed_facet_entity_counts_only_when_it_is_the_listed_company(entity, listed) -> None:
    """What landed on the rail is judged strictly: a typeahead near miss onto a
    different company with a shared word is not the competitor."""
    assert _allowlist().listed_entity(entity) == listed
