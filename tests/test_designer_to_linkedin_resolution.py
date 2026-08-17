"""Slice C.8 — designer↔LinkedIn cross-module identity adapter.

Pins the per-pair adapter at
``shared.cross_module_identity.designer_to_linkedin``:

- Portfolio URL normalization handles protocols, www, trailing slashes.
- ``match_by_behance_url`` returns IdentityMatch when LinkedIn carries
  the designer's Behance URL; ``None`` otherwise.
- ``match_by_portfolio_url`` returns IdentityMatch when both sides
  share a personal portfolio URL; ``None`` otherwise.
- ``match_by_name_and_company`` returns IdentityMatch for exact-name +
  substring-company match; ``None`` otherwise.
- ``best_match`` returns the highest-confidence match across all three.
- Confidence-band constants pin the per-pair contract.
"""

from __future__ import annotations

from shared.cross_module_identity.designer_to_linkedin import (
    BEHANCE_URL_MATCH_CONFIDENCE,
    NAME_PLUS_COMPANY_BASE_CONFIDENCE,
    NAME_PLUS_COMPANY_MAX_CONFIDENCE,
    PORTFOLIO_URL_MATCH_CONFIDENCE,
    best_match,
    match_by_behance_url,
    match_by_name_and_company,
    match_by_portfolio_url,
    normalize_portfolio_url,
)


# ---------------------------------------------------------------------------
# Confidence-band contract pins
# ---------------------------------------------------------------------------


def test_behance_url_match_confidence_is_auto_strong() -> None:
    assert BEHANCE_URL_MATCH_CONFIDENCE >= 0.85


def test_portfolio_url_match_confidence_is_auto_strong() -> None:
    assert PORTFOLIO_URL_MATCH_CONFIDENCE >= 0.85


def test_name_plus_company_band_is_auto_medium() -> None:
    assert 0.60 <= NAME_PLUS_COMPANY_BASE_CONFIDENCE <= 0.80
    assert NAME_PLUS_COMPANY_MAX_CONFIDENCE <= 0.85


# ---------------------------------------------------------------------------
# normalize_portfolio_url
# ---------------------------------------------------------------------------


def test_normalize_strips_protocol() -> None:
    assert normalize_portfolio_url("https://cargo.site/janedoe") == "cargo.site/janedoe"
    assert normalize_portfolio_url("http://cargo.site/janedoe") == "cargo.site/janedoe"


def test_normalize_strips_www() -> None:
    assert normalize_portfolio_url("https://www.janedoe.com") == "janedoe.com"


def test_normalize_strips_trailing_slash() -> None:
    assert normalize_portfolio_url("https://janedoe.com/") == "janedoe.com"


def test_normalize_strips_query_and_fragment() -> None:
    assert normalize_portfolio_url("https://janedoe.com/work?ref=behance#top") == "janedoe.com/work"


def test_normalize_lowercases_hostname() -> None:
    """Hostname is lowercased; path preserves case (URL paths are case-sensitive)."""
    assert normalize_portfolio_url("https://JaneDoe.COM/Work") == "janedoe.com/Work"


def test_normalize_empty_returns_empty() -> None:
    assert normalize_portfolio_url("") == ""
    assert normalize_portfolio_url("   ") == ""


def test_normalize_non_string_returns_empty() -> None:
    assert normalize_portfolio_url(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# match_by_behance_url
# ---------------------------------------------------------------------------


def test_behance_url_match_when_linkedin_lists_behance() -> None:
    match = match_by_behance_url(
        "https://www.behance.net/janedoe",
        ["https://behance.net/janedoe", "https://linkedin.com/in/janedoe"],
    )
    assert match is not None
    assert match.method == "behance_url_in_linkedin"
    assert match.confidence == BEHANCE_URL_MATCH_CONFIDENCE


def test_behance_url_no_match_when_different_username() -> None:
    match = match_by_behance_url(
        "https://www.behance.net/janedoe",
        ["https://behance.net/someoneelse"],
    )
    assert match is None


def test_behance_url_no_match_when_empty_designer_url() -> None:
    assert match_by_behance_url("", ["https://behance.net/janedoe"]) is None


def test_behance_url_no_match_when_empty_linkedin_urls() -> None:
    assert match_by_behance_url("https://behance.net/janedoe", []) is None


def test_behance_url_case_insensitive() -> None:
    match = match_by_behance_url(
        "https://www.behance.net/JaneDoe",
        ["https://behance.net/janedoe"],
    )
    assert match is not None


# ---------------------------------------------------------------------------
# match_by_portfolio_url
# ---------------------------------------------------------------------------


def test_portfolio_url_match_same_personal_site() -> None:
    match = match_by_portfolio_url(
        ["https://www.janedoe.com/"],
        ["http://janedoe.com"],
    )
    assert match is not None
    assert match.method == "portfolio_url_match"
    assert match.confidence == PORTFOLIO_URL_MATCH_CONFIDENCE


def test_portfolio_url_no_overlap() -> None:
    assert match_by_portfolio_url(
        ["https://cargo.site/janedoe"],
        ["https://dribbble.com/someoneelse"],
    ) is None


def test_portfolio_url_empty_lists() -> None:
    assert match_by_portfolio_url([], ["https://janedoe.com"]) is None
    assert match_by_portfolio_url(["https://janedoe.com"], []) is None
    assert match_by_portfolio_url([], []) is None


# ---------------------------------------------------------------------------
# match_by_name_and_company
# ---------------------------------------------------------------------------


def test_name_and_company_substring_match() -> None:
    match = match_by_name_and_company(
        designer_name="jane doe",
        designer_headline="Senior Designer at Linear",
        linkedin_name="jane doe",
        linkedin_company="Linear",
    )
    assert match is not None
    assert match.method == "name_plus_company"
    assert match.confidence == NAME_PLUS_COMPANY_BASE_CONFIDENCE


def test_name_and_company_strips_inc_suffix() -> None:
    match = match_by_name_and_company(
        designer_name="jane doe",
        designer_headline="Designer at Stripe",
        linkedin_name="jane doe",
        linkedin_company="Stripe, Inc.",
    )
    assert match is not None


def test_name_and_company_returns_none_on_name_mismatch() -> None:
    assert match_by_name_and_company(
        designer_name="jane doe",
        designer_headline="Designer at Linear",
        linkedin_name="john smith",
        linkedin_company="Linear",
    ) is None


def test_name_and_company_returns_none_on_empty_company() -> None:
    assert match_by_name_and_company(
        designer_name="jane doe",
        designer_headline="",
        linkedin_name="jane doe",
        linkedin_company="Linear",
    ) is None
    assert match_by_name_and_company(
        designer_name="jane doe",
        designer_headline="Designer at Linear",
        linkedin_name="jane doe",
        linkedin_company="",
    ) is None


def test_name_and_company_returns_none_on_disjoint_companies() -> None:
    assert match_by_name_and_company(
        designer_name="jane doe",
        designer_headline="Designer at Linear",
        linkedin_name="jane doe",
        linkedin_company="Figma",
    ) is None


# ---------------------------------------------------------------------------
# best_match
# ---------------------------------------------------------------------------


def test_best_match_returns_highest_confidence() -> None:
    """When Behance URL matches, returns that (0.95) even if portfolio also matches."""
    match = best_match(
        designer_profile_url="https://behance.net/janedoe",
        designer_social_urls=["https://janedoe.com"],
        designer_name="jane doe",
        designer_headline="Designer at Linear",
        linkedin_urls=["https://behance.net/janedoe", "https://janedoe.com"],
        linkedin_name="jane doe",
        linkedin_company="Linear",
    )
    assert match is not None
    assert match.method == "behance_url_in_linkedin"
    assert match.confidence == BEHANCE_URL_MATCH_CONFIDENCE


def test_best_match_falls_through_to_portfolio() -> None:
    """When Behance URL doesn't match but portfolio does, returns portfolio match."""
    match = best_match(
        designer_profile_url="https://behance.net/janedoe",
        designer_social_urls=["https://janedoe.com"],
        designer_name="jane doe",
        designer_headline="Designer at Linear",
        linkedin_urls=["https://janedoe.com"],
        linkedin_name="jane doe",
        linkedin_company="Linear",
    )
    assert match is not None
    assert match.method == "portfolio_url_match"


def test_best_match_falls_through_to_name_company() -> None:
    """When no URL matches but name+company does, returns that."""
    match = best_match(
        designer_profile_url="https://behance.net/janedoe",
        designer_social_urls=[],
        designer_name="jane doe",
        designer_headline="Designer at Linear",
        linkedin_urls=[],
        linkedin_name="jane doe",
        linkedin_company="Linear",
    )
    assert match is not None
    assert match.method == "name_plus_company"


def test_best_match_returns_none_when_nothing_matches() -> None:
    assert best_match(
        designer_profile_url="https://behance.net/janedoe",
        designer_social_urls=["https://janedoe.com"],
        designer_name="jane doe",
        designer_headline="Designer at Linear",
        linkedin_urls=["https://someoneelse.com"],
        linkedin_name="john smith",
        linkedin_company="Figma",
    ) is None
