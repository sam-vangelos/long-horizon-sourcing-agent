"""Tests for the bio/README LinkedIn-URL extraction (OSS Maintainers Slice 8).

Covers:

- The :func:`shared.contact_discovery._extract_linkedin_url` regex:
  full URL match required; trailing punctuation trimmed; bare keyword
  ("ex-LinkedIn") does not match.
- :func:`shared.contact_discovery.merge_profile_contact` resolution
  order (blog > bio > readme); provenance label
  (``contact.linkedin_url_source``) reflects which source won;
  defensive when all sources empty.
"""

from __future__ import annotations

from github.schemas import ContactInfo
from shared.contact_discovery import _extract_linkedin_url, merge_profile_contact


# ---------------------------------------------------------------------------
# _extract_linkedin_url — regex contract
# ---------------------------------------------------------------------------


def test_extracts_full_personal_profile_url() -> None:
    text = "Find me at https://linkedin.com/in/jane-doe-123 if interested."
    assert (
        _extract_linkedin_url(text) == "https://linkedin.com/in/jane-doe-123"
    )


def test_extracts_www_subdomain() -> None:
    text = "https://www.linkedin.com/in/janedoe is my profile"
    assert _extract_linkedin_url(text) == "https://www.linkedin.com/in/janedoe"


def test_extracts_company_url() -> None:
    """We capture company URLs too; the resolver chooses whether to use them."""

    text = "Worked at https://linkedin.com/company/example-corp"
    assert (
        _extract_linkedin_url(text)
        == "https://linkedin.com/company/example-corp"
    )


def test_does_not_match_bare_keyword() -> None:
    """Spec §12 false-positive mitigation: 'ex-LinkedIn' must not match."""

    assert _extract_linkedin_url("ex-LinkedIn engineer") == ""
    assert _extract_linkedin_url("Used to work at LinkedIn") == ""


def test_does_not_match_partial_url() -> None:
    """A bare host without a path is not a profile."""

    assert _extract_linkedin_url("linkedin.com is the host") == ""


def test_strips_trailing_punctuation() -> None:
    text = "Profile: https://linkedin.com/in/jane-doe."
    assert _extract_linkedin_url(text) == "https://linkedin.com/in/jane-doe"


def test_returns_first_match_only() -> None:
    text = (
        "https://linkedin.com/in/first "
        "and https://linkedin.com/in/second"
    )
    assert _extract_linkedin_url(text) == "https://linkedin.com/in/first"


def test_empty_input_returns_empty() -> None:
    assert _extract_linkedin_url("") == ""
    assert _extract_linkedin_url(None or "") == ""


# ---------------------------------------------------------------------------
# merge_profile_contact — resolution order + provenance
# ---------------------------------------------------------------------------


def test_blog_wins_over_bio_and_readme() -> None:
    """Blog field is highest confidence and always wins when present."""

    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="https://linkedin.com/in/from-blog",
        bio="Also at https://linkedin.com/in/from-bio",
        readme_text="Or https://linkedin.com/in/from-readme",
    )
    assert contact.linkedin_url == "https://linkedin.com/in/from-blog"
    assert contact.linkedin_url_source == "blog"


def test_bio_used_when_blog_is_not_linkedin() -> None:
    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="https://example.com/personal",
        bio="LinkedIn: https://linkedin.com/in/from-bio",
        readme_text="",
    )
    assert contact.linkedin_url == "https://linkedin.com/in/from-bio"
    assert contact.linkedin_url_source == "bio"


def test_readme_used_when_blog_and_bio_empty() -> None:
    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="",
        bio="",
        readme_text="Connect at https://linkedin.com/in/from-readme",
    )
    assert contact.linkedin_url == "https://linkedin.com/in/from-readme"
    assert contact.linkedin_url_source == "readme"


def test_no_linkedin_url_when_all_sources_empty() -> None:
    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="alice@example.com",
        twitter="alice",
        blog="https://alice.dev",
        bio="ML engineer interested in distributed systems",
        readme_text="See my repos for code samples",
    )
    assert contact.linkedin_url == ""
    assert contact.linkedin_url_source == ""
    # Other contact fields still merged correctly.
    assert "alice@example.com" in contact.emails
    assert contact.twitter_url == "https://twitter.com/alice"
    assert contact.website == "https://alice.dev"


def test_blog_field_with_non_url_value_falls_back_to_bio() -> None:
    """Non-LinkedIn blog (e.g., medium.com) doesn't claim the field."""

    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="medium.com/@alice",  # not LinkedIn
        bio="LinkedIn: https://linkedin.com/in/alice",
        readme_text="",
    )
    assert contact.linkedin_url == "https://linkedin.com/in/alice"
    assert contact.linkedin_url_source == "bio"
    # The non-LinkedIn blog still becomes the website.
    assert contact.website == "https://medium.com/@alice"


def test_bio_with_only_keyword_does_not_set_linkedin_url() -> None:
    """Spec §12: bare 'LinkedIn' keyword must not falsely cross-link."""

    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="",
        bio="Ex-LinkedIn engineer; now working on training infra.",
        readme_text="",
    )
    assert contact.linkedin_url == ""
    assert contact.linkedin_url_source == ""


def test_backwards_compat_call_without_bio_or_readme() -> None:
    """Existing callers (pre-Slice-8) pass only positional args; behavior unchanged."""

    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        "",
        "",
        "https://linkedin.com/in/legacy",
    )
    assert contact.linkedin_url == "https://linkedin.com/in/legacy"
    assert contact.linkedin_url_source == "blog"


# ---------------------------------------------------------------------------
# P3.8 — company-page URLs must never be assigned as contact.linkedin_url.
#
# `_extract_linkedin_url`'s regex deliberately captures `linkedin.com/company/`
# pages too (see `test_extracts_company_url` above — the parser keeps them
# so the caller can log discoveries it chooses not to use), but a company
# page is not a person, so it must never win the linkedin_url/linkedin_url_source
# assignment in `merge_profile_contact` — from the blog field, from bio prose,
# or from a profile README.
# ---------------------------------------------------------------------------


def test_company_url_in_blog_not_assigned_as_linkedin_url() -> None:
    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="https://linkedin.com/company/example-corp",
        bio="",
        readme_text="",
    )
    assert contact.linkedin_url == ""
    assert contact.linkedin_url_source == ""
    # The blog is still recorded as the candidate's website — only the
    # linkedin_url/linkedin_url_source assignment is filtered.
    assert contact.website == "https://linkedin.com/company/example-corp"


def test_company_url_in_bio_not_assigned_as_linkedin_url() -> None:
    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="",
        bio="Formerly at https://linkedin.com/company/example-corp",
        readme_text="",
    )
    assert contact.linkedin_url == ""
    assert contact.linkedin_url_source == ""


def test_company_url_in_readme_not_assigned_as_linkedin_url() -> None:
    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="",
        bio="",
        readme_text="Team page: https://linkedin.com/company/example-corp",
    )
    assert contact.linkedin_url == ""
    assert contact.linkedin_url_source == ""


def test_personal_profile_url_still_works_alongside_company_filter() -> None:
    """Regression: filtering company URLs must not break the personal /in/ path."""

    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="https://linkedin.com/in/real-person",
        bio="",
        readme_text="",
    )
    assert contact.linkedin_url == "https://linkedin.com/in/real-person"
    assert contact.linkedin_url_source == "blog"


def test_company_url_in_bio_falls_through_to_personal_readme_url() -> None:
    """A rejected company URL must not block a later, personal fallback source."""

    contact = ContactInfo()
    contact = merge_profile_contact(
        contact,
        user_email="",
        twitter="",
        blog="",
        bio="https://linkedin.com/company/example-corp",
        readme_text="Personal: https://linkedin.com/in/real-person",
    )
    assert contact.linkedin_url == "https://linkedin.com/in/real-person"
    assert contact.linkedin_url_source == "readme"
