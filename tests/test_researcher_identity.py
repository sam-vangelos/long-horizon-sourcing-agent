"""Researcher Slice 4 — disambiguation coverage.

Asserts:
- Country filter drops candidates whose affiliations don't carry an
  allowed ISO code (defensive against affiliation-string elision).
- Concept-overlap filter drops candidates whose x_concepts don't
  intersect the required set.
- papers_in_window_floor drops candidates below the floor.
- Common-name collisions among ORCID-less candidates flag both for
  manual review (per Spec Opinion 3 — INFERENTIAL_SAVE routing).
- ORCID-anchored candidates with the same display name are NOT
  flagged (ORCID disambiguates them).
- The disambiguation summary counters match the operations performed.
"""

from __future__ import annotations

from researcher.identity import disambiguate
from researcher.schemas import ResearcherCandidate


def _candidate(
    *,
    author_id: str,
    name: str,
    orcid: str = "",
    affiliations: list[str] | None = None,
    papers_in_window: int = 5,
    x_concept_ids: list[str] | None = None,
    country_codes_in_raw: list[str] | None = None,
) -> ResearcherCandidate:
    """Build a ResearcherCandidate with a synthetic raw_openalex
    payload so the country/concept filters have something to match
    against.
    """

    raw_concepts = [
        {"id": f"https://openalex.org/{cid}"} for cid in (x_concept_ids or [])
    ]
    raw_institutions = [
        {"display_name": "Inst", "country_code": code}
        for code in (country_codes_in_raw or [])
    ]
    return ResearcherCandidate(
        author_id=author_id,
        orcid=orcid,
        name=name,
        affiliations=affiliations or [],
        papers_in_window=papers_in_window,
        raw_openalex={
            "x_concepts": raw_concepts,
            "last_known_institutions": raw_institutions,
        },
    )


def test_country_filter_drops_candidates_outside_allowed_set() -> None:
    candidates = [
        _candidate(author_id="A1", name="In US", affiliations=["MIT (US)"]),
        _candidate(author_id="A2", name="In FR", affiliations=["INRIA (FR)"]),
    ]
    summary = disambiguate(candidates, allowed_country_codes=["US", "GB"])
    assert summary.kept_count == 1
    assert summary.dropped_country == 1
    kept = [r for r in summary.results if r.kept]
    assert kept[0].candidate.author_id == "A1"


def test_country_filter_falls_back_to_raw_payload_country_codes() -> None:
    """If the affiliation string elided the country code, the raw
    OpenAlex payload's `last_known_institutions[*].country_code` is
    consulted as a fallback."""

    candidates = [
        _candidate(
            author_id="A1",
            name="X",
            affiliations=["MIT"],  # No (US) suffix
            country_codes_in_raw=["US"],
        ),
    ]
    summary = disambiguate(candidates, allowed_country_codes=["US"])
    assert summary.kept_count == 1


def test_concept_filter_drops_candidates_without_overlap() -> None:
    candidates = [
        _candidate(
            author_id="A1",
            name="NLP author",
            x_concept_ids=["C2778407487"],
            affiliations=["MIT (US)"],
        ),
        _candidate(
            author_id="A2",
            name="Vision author",
            x_concept_ids=["C9999"],
            affiliations=["MIT (US)"],
        ),
    ]
    summary = disambiguate(
        candidates,
        required_concept_ids=["C2778407487"],
    )
    assert summary.kept_count == 1
    assert summary.dropped_concept == 1


def test_papers_in_window_floor_drops_low_count_candidates() -> None:
    candidates = [
        _candidate(author_id="A1", name="High", papers_in_window=10),
        _candidate(author_id="A2", name="Low", papers_in_window=1),
    ]
    summary = disambiguate(candidates, papers_in_window_floor=3)
    assert summary.kept_count == 1
    assert summary.dropped_papers_in_window == 1


def test_common_name_collision_flags_both_orcid_less_candidates() -> None:
    """Two ORCID-less authors with the same name should both be flagged
    for manual review (INFERENTIAL_SAVE routing per Spec Opinion 3).
    """

    candidates = [
        _candidate(author_id="A1", name="Wei Wang", affiliations=["MIT (US)"]),
        _candidate(author_id="A2", name="Wei Wang", affiliations=["Stanford (US)"]),
    ]
    summary = disambiguate(candidates, allowed_country_codes=["US"])
    assert summary.kept_count == 2
    assert summary.flagged_common_name == 2
    for result in summary.results:
        if result.kept:
            assert result.needs_manual_review is True
            assert "common_name_collision" in result.review_note


def test_orcid_anchored_candidates_with_same_name_are_not_flagged() -> None:
    """ORCID disambiguates: if any in the colliding group has an ORCID,
    only the ORCID-less ones are ambiguous.
    """

    candidates = [
        _candidate(
            author_id="A1",
            name="Wei Wang",
            orcid="0000-0001-2345-6789",
            affiliations=["MIT (US)"],
        ),
        _candidate(
            author_id="A2",
            name="Wei Wang",
            orcid="0000-0002-3456-7890",
            affiliations=["Stanford (US)"],
        ),
    ]
    summary = disambiguate(candidates, allowed_country_codes=["US"])
    assert summary.kept_count == 2
    assert summary.flagged_common_name == 0


def test_no_identity_anchor_drops_candidate() -> None:
    """A candidate with neither ORCID nor author_id should be dropped
    cleanly with a `no_identity_key` rejection reason.
    """

    candidates = [
        ResearcherCandidate(author_id="", orcid="", name="No Anchor"),
        ResearcherCandidate(author_id="A1", orcid="", name="OK"),
    ]
    summary = disambiguate(candidates)
    assert summary.kept_count == 1
    assert summary.dropped_missing_identity == 1
    rejected = [r for r in summary.results if not r.kept]
    assert rejected[0].rejection_reason == "no_identity_key"


def test_summary_counters_align_with_operations() -> None:
    candidates = [
        _candidate(
            author_id="A1",
            name="Ok",
            affiliations=["MIT (US)"],
            x_concept_ids=["C1"],
            papers_in_window=10,
        ),
        _candidate(
            author_id="A2",
            name="Wrong country",
            affiliations=["INRIA (FR)"],
        ),
        _candidate(
            author_id="A3",
            name="Wrong concept",
            affiliations=["MIT (US)"],
            x_concept_ids=["C9999"],
        ),
        _candidate(
            author_id="A4",
            name="Too few",
            affiliations=["MIT (US)"],
            x_concept_ids=["C1"],
            papers_in_window=1,
        ),
    ]
    summary = disambiguate(
        candidates,
        allowed_country_codes=["US"],
        required_concept_ids=["C1"],
        papers_in_window_floor=5,
    )
    assert summary.input_count == 4
    assert summary.kept_count == 1
    assert summary.dropped_country == 1
    assert summary.dropped_concept == 1
    assert summary.dropped_papers_in_window == 1
    assert summary.dropped_missing_identity == 0
