"""Slice B.3 — researcher↔LinkedIn cross-module identity adapter.

Pins the per-pair adapter at
``shared.cross_module_identity.researcher_to_linkedin``:

- ORCID normalization handles URL prefixes + checksum-X cases.
- ``match_by_orcid`` returns IdentityMatch for matching ORCIDs;
  ``None`` for empty / malformed / mismatched.
- ``match_by_name_and_affiliation`` returns IdentityMatch for
  exact-name + substring-affiliation match; ``None`` otherwise.
- Confidence-band constants pin the per-pair contract.

The full integration into
``shared.identity_resolution_service``'s pass structure is a
behavior-preserving follow-up; this slice ships the adapter
primitives the resolver will compose.
"""

from __future__ import annotations

from shared.cross_module_identity.researcher_to_linkedin import (
    IdentityMatch,
    NAME_PLUS_AFFILIATION_BASE_CONFIDENCE,
    NAME_PLUS_AFFILIATION_MAX_CONFIDENCE,
    NAME_PLUS_PUBLICATION_BASE_CONFIDENCE,
    NAME_PLUS_PUBLICATION_MAX_CONFIDENCE,
    ORCID_MATCH_CONFIDENCE,
    match_by_name_and_affiliation,
    match_by_orcid,
    normalize_orcid,
)


# ---------------------------------------------------------------------------
# Confidence-band contract pins
# ---------------------------------------------------------------------------


def test_orcid_match_confidence_is_auto_strong() -> None:
    """ORCID match clears the auto-strong threshold (≥0.85)."""

    assert ORCID_MATCH_CONFIDENCE >= 0.85


def test_name_plus_affiliation_band_is_auto_medium() -> None:
    """Name+affiliation match lands in the auto-medium band (0.65-0.85)."""

    assert 0.65 <= NAME_PLUS_AFFILIATION_BASE_CONFIDENCE <= 0.85
    assert NAME_PLUS_AFFILIATION_MAX_CONFIDENCE <= 0.85


def test_name_plus_publication_band_is_pending_decision() -> None:
    """Name+publication match lands in the pending-decision band (≤0.75)."""

    assert NAME_PLUS_PUBLICATION_BASE_CONFIDENCE >= 0.60
    assert NAME_PLUS_PUBLICATION_MAX_CONFIDENCE <= 0.80


# ---------------------------------------------------------------------------
# normalize_orcid
# ---------------------------------------------------------------------------


def test_normalize_orcid_canonical_form() -> None:
    assert normalize_orcid("0000-0002-1825-0097") == "0000-0002-1825-0097"


def test_normalize_orcid_strips_url_prefix() -> None:
    for prefix in (
        "https://orcid.org/",
        "http://orcid.org/",
        "orcid.org/",
    ):
        assert (
            normalize_orcid(f"{prefix}0000-0002-1825-0097")
            == "0000-0002-1825-0097"
        )


def test_normalize_orcid_handles_checksum_x() -> None:
    """The trailing checksum can be 'X' for some ORCIDs."""

    assert normalize_orcid("0000-0001-5109-3700") == "0000-0001-5109-3700"
    assert normalize_orcid("0000-0001-2345-678X") == "0000-0001-2345-678X"


def test_normalize_orcid_strips_whitespace() -> None:
    assert normalize_orcid("  0000-0002-1825-0097  ") == "0000-0002-1825-0097"


def test_normalize_orcid_rejects_malformed() -> None:
    for bad in (
        "",
        "0000-0002-1825",
        "0000_0002_1825_0097",
        "abcd-0002-1825-0097",
        "not an orcid",
    ):
        assert normalize_orcid(bad) == "", f"expected empty for {bad!r}"


def test_normalize_orcid_rejects_non_string() -> None:
    assert normalize_orcid(None) == ""  # type: ignore[arg-type]
    assert normalize_orcid(12345) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# match_by_orcid
# ---------------------------------------------------------------------------


def test_match_by_orcid_returns_identity_match_when_both_present() -> None:
    match = match_by_orcid("0000-0002-1825-0097", "0000-0002-1825-0097")
    assert match is not None
    assert match.method == "orcid_match"
    assert match.confidence == ORCID_MATCH_CONFIDENCE
    assert "0000-0002-1825-0097" in match.evidence[0]


def test_match_by_orcid_handles_url_prefix_on_either_side() -> None:
    match = match_by_orcid(
        "https://orcid.org/0000-0002-1825-0097",
        "0000-0002-1825-0097",
    )
    assert match is not None
    assert match.method == "orcid_match"


def test_match_by_orcid_returns_none_when_empty() -> None:
    assert match_by_orcid("", "0000-0002-1825-0097") is None
    assert match_by_orcid("0000-0002-1825-0097", "") is None
    assert match_by_orcid("", "") is None


def test_match_by_orcid_returns_none_on_mismatch() -> None:
    assert (
        match_by_orcid(
            "0000-0002-1825-0097",
            "0000-0001-5109-3700",
        )
        is None
    )


def test_match_by_orcid_returns_none_on_malformed() -> None:
    assert match_by_orcid("not an orcid", "0000-0002-1825-0097") is None


# ---------------------------------------------------------------------------
# match_by_name_and_affiliation
# ---------------------------------------------------------------------------


def test_match_by_name_and_affiliation_substring_overlap() -> None:
    """Researcher's clean affiliation is a substring of LinkedIn's noisy one."""

    match = match_by_name_and_affiliation(
        researcher_name="jane doe",
        researcher_affiliation="Stanford University",
        linkedin_name="jane doe",
        linkedin_company="Stanford University - Computer Science",
    )
    assert match is not None
    assert match.method == "name_plus_affiliation"
    assert match.confidence == NAME_PLUS_AFFILIATION_BASE_CONFIDENCE


def test_match_by_name_and_affiliation_strips_corporate_suffixes() -> None:
    """Common corp suffixes ('Inc', 'LLC') get stripped before substring check."""

    match = match_by_name_and_affiliation(
        researcher_name="jane doe",
        researcher_affiliation="Google",
        linkedin_name="jane doe",
        linkedin_company="Google, LLC",
    )
    assert match is not None
    assert match.method == "name_plus_affiliation"


def test_match_by_name_and_affiliation_returns_none_when_names_differ() -> None:
    assert (
        match_by_name_and_affiliation(
            researcher_name="jane doe",
            researcher_affiliation="Stanford",
            linkedin_name="john smith",
            linkedin_company="Stanford",
        )
        is None
    )


def test_match_by_name_and_affiliation_returns_none_when_affiliations_disjoint() -> None:
    assert (
        match_by_name_and_affiliation(
            researcher_name="jane doe",
            researcher_affiliation="Stanford",
            linkedin_name="jane doe",
            linkedin_company="MIT CSAIL",
        )
        is None
    )


def test_match_by_name_and_affiliation_returns_none_on_empty() -> None:
    assert (
        match_by_name_and_affiliation(
            researcher_name="",
            researcher_affiliation="Stanford",
            linkedin_name="jane doe",
            linkedin_company="Stanford",
        )
        is None
    )
    assert (
        match_by_name_and_affiliation(
            researcher_name="jane doe",
            researcher_affiliation="",
            linkedin_name="jane doe",
            linkedin_company="Stanford",
        )
        is None
    )
