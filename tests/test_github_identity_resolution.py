"""Tests for OSS Maintainers Slice 8 — github branch of `_extract_signals`.

Pins the contract: the resolver reads ``contact.linkedin_url`` and
``contact.linkedin_url_source`` from the github terminal payload and
populates ``candidate.linkedin_handle`` + ``candidate.linkedin_url_source``.
The provenance label ("blog" / "bio" / "readme") flows through so
downstream confidence-banding consumers can treat bio/readme matches
as lower-confidence than blog matches.
"""

from __future__ import annotations

from shared.identity_resolution_service import _Candidate, _extract_signals


def _make_github_candidate(
    *,
    display_name: str = "alice",
    contact: dict | None = None,
    user: dict | None = None,
) -> _Candidate:
    return _Candidate(
        source="github",
        state_key="brief_x",
        candidate_id=1,
        display_name=display_name,
        profile_url=f"https://github.com/{display_name}",
        terminal_payload={
            "candidate_record": {
                "user": user or {"name": "Alice Doe"},
                "contact": contact or {},
            }
        },
    )


def test_blog_provenance_propagates() -> None:
    candidate = _make_github_candidate(
        contact={
            "linkedin_url": "https://linkedin.com/in/alice-blog",
            "linkedin_url_source": "blog",
        }
    )

    _extract_signals(candidate)

    assert candidate.linkedin_handle == "alice-blog"
    assert candidate.linkedin_url_source == "blog"
    assert candidate.real_name == "Alice Doe"


def test_bio_provenance_propagates() -> None:
    candidate = _make_github_candidate(
        contact={
            "linkedin_url": "https://linkedin.com/in/alice-bio",
            "linkedin_url_source": "bio",
        }
    )

    _extract_signals(candidate)

    assert candidate.linkedin_handle == "alice-bio"
    assert candidate.linkedin_url_source == "bio"


def test_readme_provenance_propagates() -> None:
    candidate = _make_github_candidate(
        contact={
            "linkedin_url": "https://linkedin.com/in/alice-readme",
            "linkedin_url_source": "readme",
        }
    )

    _extract_signals(candidate)

    assert candidate.linkedin_handle == "alice-readme"
    assert candidate.linkedin_url_source == "readme"


def test_no_provenance_when_handle_unresolved() -> None:
    """Provenance label only sticks when the handle resolved to non-empty."""

    candidate = _make_github_candidate(
        contact={"linkedin_url": "", "linkedin_url_source": "bio"}
    )

    _extract_signals(candidate)

    assert candidate.linkedin_handle == ""
    assert candidate.linkedin_url_source == ""


def test_legacy_payloads_without_source_still_resolve() -> None:
    """Pre-Slice-8 payloads have no ``linkedin_url_source`` field; default empty."""

    candidate = _make_github_candidate(
        contact={"linkedin_url": "https://linkedin.com/in/alice-legacy"}
    )

    _extract_signals(candidate)

    assert candidate.linkedin_handle == "alice-legacy"
    # No provenance recorded — empty string, not a crash.
    assert candidate.linkedin_url_source == ""


def test_top_level_linkedin_url_hint_falls_back_when_contact_empty() -> None:
    """Existing fallback path (matched_profile_url etc.) still works."""

    candidate = _Candidate(
        source="github",
        state_key="brief_x",
        candidate_id=2,
        display_name="bob",
        profile_url="https://github.com/bob",
        terminal_payload={
            "candidate_record": {
                "user": {"name": "Bob Smith"},
                "matched_profile_url": "https://linkedin.com/in/bob-matched",
            }
        },
    )

    _extract_signals(candidate)

    assert candidate.linkedin_handle == "bob-matched"
    # No provenance because the URL came from a fallback key, not contact.
    assert candidate.linkedin_url_source == ""


def test_linkedin_source_unaffected() -> None:
    """A LinkedIn-source candidate doesn't carry github provenance."""

    candidate = _Candidate(
        source="linkedin",
        state_key="brief_x",
        candidate_id=3,
        display_name="Carol Jones",
        profile_url="https://linkedin.com/in/carol-jones",
        terminal_payload={},
    )

    _extract_signals(candidate)

    assert candidate.linkedin_handle == "carol-jones"
    assert candidate.linkedin_url_source == ""
