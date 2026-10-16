"""Exa source module S2 — intermediate schema coverage.

The load-bearing properties are identity (an Exa find and a LinkedIn
find of the same person must produce the same dedupe key), evidence
(every retrieved URL reaches the judge), and the verify-before-outreach
flag being STRUCTURAL — present or absent as a block, not a sentence
somewhere in prose that a downstream surface would have to grep for.
"""

from __future__ import annotations

import json

from exa.linkedin_render import parse_exa_linkedin_markdown
from exa.schemas import (
    EVIDENCE_TEXT_EXCERPT_CAP,
    TENURE_LABEL,
    ExaCandidate,
    ExaEvidence,
    ExaProgress,
    ExaSearchQuery,
    derive_identity_key,
)
from shared.identity_resolution import normalize_public_linkedin_handle
from shared.judger import _profile_to_text


# Exa's LinkedIn profile rendering — the live shape, synthetic name.
_RENDER_PAGE = """# Avery Holt

Investor @ Benchmark

San Francisco Bay Area (US)

500 connections • 2,111 followers

## About

averyholt.xyz

Total Experience: 4 years and 11 months

## Experience

### Investor - [Benchmark](https://www.linkedin.com/company/benchmark-vc) (Current)

Jan 2025 - Present (1 year and 7 months) in San Francisco, California, United States

Department: Consulting

Benchmark is a Venture Capital and Private Equity Principals company with 40-50 employees.

### Member of Technical Staff - [Tensor Harbor](https://www.linkedin.com/company/tensor-harbor)

Jan 2023 - Jan 2024 (1 year) in New York, United States

Department: Engineering and Technical

## Education

### BS at Stanford University

1997 - 1999 (2 years)
"""


def _evidence() -> list[ExaEvidence]:
    return [
        ExaEvidence(
            url="https://www.linkedin.com/in/averylinholt",
            title="Avery Holt — Investor, Benchmark",
            published_date="2026-06-01",
            highlights=["Member of Technical Staff at Modal before investing"],
            text_excerpt="Benchmark's one junior investor.",
            retrieved_at="2026-08-28T00:00:00+00:00",
        ),
        ExaEvidence(
            url="https://averyholt.xyz",
            title="Avery Holt",
            published_date="",
            highlights=["straddling product growth and engineering"],
            text_excerpt="Personal site.",
            retrieved_at="2026-08-28T00:00:00+00:00",
        ),
    ]


def _candidate(**overrides) -> ExaCandidate:
    payload = dict(
        identity_key="linkedin.com/in/averylinholt",
        name="Avery Holt",
        current_title="Investor",
        current_firm="Benchmark",
        location="San Francisco",
        linkedin_url="https://www.linkedin.com/in/averylinholt",
        evidence=_evidence(),
        source_query_id=1,
        source_lane="people_search",
    )
    payload.update(overrides)
    return ExaCandidate(**payload)


# ---------------------------------------------------------------------------
# derive_identity_key
# ---------------------------------------------------------------------------


def test_derive_identity_key_prefers_linkedin_path() -> None:
    assert (
        derive_identity_key(
            "Avery Holt", "https://www.linkedin.com/in/Averylinholt/", "Benchmark"
        )
        == "linkedin.com/in/averylinholt"
    )


def test_derive_identity_key_normalizes_url_variation() -> None:
    """Scheme, www, trailing slash, query string and case are not identity."""

    variants = [
        "https://www.linkedin.com/in/Averylinholt/",
        "http://linkedin.com/in/averylinholt",
        "linkedin.com/in/averylinholt?trk=public_profile",
        "https://www.linkedin.com/in/averylinholt/en",
    ]
    keys = {derive_identity_key("Avery Holt", v, "Benchmark") for v in variants}
    assert keys == {"linkedin.com/in/averylinholt"}


def test_derive_identity_key_accepts_relative_in_path() -> None:
    """Team-page anchors are often relative — `/in/x`, not a full URL."""

    assert (
        derive_identity_key("Avery Holt", "/in/Averylinholt", "Benchmark")
        == "linkedin.com/in/averylinholt"
    )


def test_derive_identity_key_matches_shared_handle_normalizer() -> None:
    """The key must be derived by the matcher's own rules, not a local regex.

    A key normalized differently from
    `shared.identity_resolution.normalize_public_linkedin_handle` is a
    key that silently fails to converge with a LinkedIn find of the
    same person.
    """

    for url in (
        "https://www.linkedin.com/in/Averylinholt/",
        "/in/averylinholt",
        "https://uk.linkedin.com/in/averylinholt",
        "linkedin.com/in/avery-lin-holt",
    ):
        handle = normalize_public_linkedin_handle(url)
        assert handle
        assert derive_identity_key("Avery Holt", url, "Benchmark") == (
            f"linkedin.com/in/{handle}"
        )

    # A recruiter URL carries no public handle — it must not become a key.
    recruiter_url = "https://www.linkedin.com/talent/profile/ABC123"
    assert normalize_public_linkedin_handle(recruiter_url) == ""
    assert derive_identity_key("Avery Holt", recruiter_url, "Benchmark") == (
        "avery-holt-benchmark"
    )


def test_derive_identity_key_falls_back_to_name_firm_slug() -> None:
    assert (
        derive_identity_key("Avery Holt", "", "Benchmark") == "avery-holt-benchmark"
    )
    # A firm page URL is not a profile URL — it must not become the key.
    assert (
        derive_identity_key("Theo Marsh", "https://www.kilnventures.com/team", "Kiln Ventures")
        == "theo-marsh-kiln-ventures"
    )


# ---------------------------------------------------------------------------
# ExaCandidate
# ---------------------------------------------------------------------------


def test_exa_candidate_round_trips_through_dict() -> None:
    candidate = _candidate()
    assert ExaCandidate.from_dict(candidate.to_dict()) == candidate


def test_exa_evidence_from_api_tolerates_missing_fields() -> None:
    parsed = ExaEvidence.from_api({"url": "https://example.com", "text": None})
    assert parsed.url == "https://example.com"
    assert parsed.title == ""
    assert parsed.highlights == []
    assert parsed.text_excerpt == ""
    assert ExaEvidence.from_api(None).url == ""


def test_evidence_text_excerpt_is_capped_at_ingestion() -> None:
    evidence = ExaEvidence(url="https://example.com", text_excerpt="x" * 5000)
    assert len(evidence.text_excerpt) == EVIDENCE_TEXT_EXCERPT_CAP
    assert len(evidence.to_dict()["text_excerpt"]) == EVIDENCE_TEXT_EXCERPT_CAP


def test_exa_evidence_round_trip_is_lossless_after_ingestion_cap() -> None:
    """Capping in ``to_dict`` made save-then-load silently change the record."""

    evidence = ExaEvidence(url="https://example.com", text_excerpt="x" * 5000)
    assert ExaEvidence.from_dict(evidence.to_dict()) == evidence

    from_api = ExaEvidence.from_api({"url": "https://example.com", "text": "y" * 5000})
    assert ExaEvidence.from_dict(from_api.to_dict()) == from_api


def test_to_snippet_maps_title_firm_location() -> None:
    snippet = _candidate().to_snippet(
        source_string_id=7, source_string_name="people_search · Benchmark"
    )
    assert snippet.name == "Avery Holt"
    assert snippet.current_title == "Investor"
    assert snippet.current_company == "Benchmark"
    assert snippet.location == "San Francisco"
    assert snippet.headline == "Investor at Benchmark"
    assert snippet.profile_url == "https://www.linkedin.com/in/averylinholt"
    assert snippet.source_string_id == 7
    assert snippet.source_string_name == "people_search · Benchmark"


def test_snippet_and_summary_never_fabricate_history() -> None:
    """Exa has no career history; neither adapter may manufacture one.

    ``shared/judger.py:1881-1885`` renders ``experience_entries`` under
    a "Career History:" heading, so page titles placed there would read
    to the judge as prior roles. Likewise a team-page highlight is not
    a bullet the person wrote about their current role. Evidence
    reaches a model only through ``to_evidence_text``.
    """

    candidate = _candidate()
    snippet = candidate.to_snippet()
    assert snippet.experience_entries == []

    summary = candidate.to_profile_summary()
    assert len(summary.experiences) == 1
    experience = summary.experiences[0]
    assert experience.title == "Investor"
    assert experience.company == "Benchmark"
    assert experience.location == "San Francisco"
    assert experience.start == "" and experience.end == ""
    assert experience.summary_bullets == []
    # The highlights still reach a model — through the evidence block.
    assert (
        "Member of Technical Staff at Modal before investing"
        in candidate.to_evidence_text()
    )


def test_profile_summary_carries_dated_experiences_and_education_from_render() -> None:
    """With a LinkedIn render the summary is a real profile, not a stub.

    ``shared/judger.py:1889-1918`` prints each position as
    ``- Title at Company (start-end)`` and the About section verbatim.
    Before S7 an Exa record printed ``About: None listed`` and one
    ``( - ): no details`` line, so the full-evaluation prompt's
    sparse-profile doctrine fired on an artifact of the source.
    """

    render = parse_exa_linkedin_markdown(_RENDER_PAGE)
    assert render is not None
    candidate = _candidate(
        current_title="Investor",
        current_firm="Benchmark",
        profile_render=render.to_dict(),
    )

    summary = candidate.to_profile_summary()

    assert summary.headline == "Investor @ Benchmark"
    assert [(e.title, e.company, e.start, e.end) for e in summary.experiences] == [
        ("Investor", "Benchmark", "Jan 2025", "Present"),
        ("Member of Technical Staff", "Tensor Harbor", "Jan 2023", "Jan 2024"),
    ]
    assert [(e.degree, e.school, e.start, e.end) for e in summary.education] == [
        ("BS", "Stanford University", "1997", "1999")
    ]
    # The tenure is Exa's arithmetic over the position dates, not a
    # sentence the candidate wrote — so it gets its own labeled line
    # rather than sitting inside the person's own About prose.
    assert summary.about == (
        f"averyholt.xyz\n\n{TENURE_LABEL}: 4 years and 11 months"
    )
    assert TENURE_LABEL == "Platform-derived tenure (Exa)"
    assert "Total Experience" not in summary.about

    # The company blurb Exa appends is about the FIRM. It never becomes a
    # bullet the person wrote about their role.
    assert all(e.summary_bullets == [] for e in summary.experiences)

    # And the rendered profile text the judge actually reads is no longer
    # the sparse stub that produced the 2026-09-03 failures.
    rendered = _profile_to_text(summary)
    assert "- Investor at Benchmark (Jan 2025-Present)" in rendered
    assert "About:\naveryholt.xyz" in rendered
    assert "None listed" not in rendered


def test_profile_summary_carries_the_persons_own_position_bullets() -> None:
    """Position bullets reach the judge as ``summary_bullets``.

    The firm blurb still never does; the bullets are the person's own
    words, which is what ``summary_bullets`` means on a LinkedIn profile.
    """

    page = _RENDER_PAGE.replace(
        "Department: Engineering and Technical",
        "- Reviewed every pull request on the inference team.\n\n"
        "Department: Engineering and Technical",
    )
    render = parse_exa_linkedin_markdown(page)
    assert render is not None
    candidate = _candidate(profile_render=render.to_dict())

    summary = candidate.to_profile_summary()
    bullets = {e.title: e.summary_bullets for e in summary.experiences}
    assert bullets["Member of Technical Staff"] == [
        "Reviewed every pull request on the inference team."
    ]
    assert bullets["Investor"] == []

    rendered = _profile_to_text(summary)
    assert "Reviewed every pull request on the inference team." in rendered
    assert "40-50 employees" not in rendered


def test_profile_summary_fallback_without_render_unchanged() -> None:
    """A team-page harvest has no LinkedIn page; its summary is untouched."""

    candidate = _candidate()
    assert candidate.profile_render is None
    assert candidate.render() is None

    summary = candidate.to_profile_summary()

    assert summary.headline == "Investor at Benchmark"
    assert len(summary.experiences) == 1
    experience = summary.experiences[0]
    assert (experience.title, experience.company) == ("Investor", "Benchmark")
    assert experience.location == "San Francisco"
    assert experience.start == "" and experience.end == ""
    assert experience.summary_bullets == []
    assert summary.education == []
    assert summary.about == ""

    # An empty dict is not a render either — no half-built profile.
    assert _candidate(profile_render={}).render() is None


def test_candidate_with_render_round_trips_through_dict() -> None:
    render = parse_exa_linkedin_markdown(_RENDER_PAGE)
    assert render is not None
    candidate = _candidate(profile_render=render.to_dict())

    restored = ExaCandidate.from_dict(candidate.to_dict())

    assert restored == candidate
    assert restored.render() == render
    # JSON is the storage medium; the payload must survive it verbatim.
    assert json.loads(candidate.to_json())["profile_render"] == render.to_dict()

    # A record written before S7 carries no key at all and still loads.
    legacy = candidate.to_dict()
    legacy.pop("profile_render")
    assert ExaCandidate.from_dict(legacy).profile_render is None


def test_from_dict_derives_render_from_stored_linkedin_excerpt() -> None:
    """A row written BEFORE S7 must not be re-judged as a thin profile.

    ``exa/orchestrator.py:1535`` rebuilds every pending candidate with
    ``ExaCandidate.from_dict(entry.record)`` on resume. Those rows carry
    no ``profile_render`` key — but they DO carry the LinkedIn page,
    truncated at the old 2000-character cap, in
    ``evidence[].text_excerpt``.
    """

    # The shape the live store actually holds: no profile_render key,
    # blank title/firm/location, insufficient, one capped excerpt.
    stored = {
        "identity_key": "linkedin.com/in/averylinholt",
        "name": "Avery Holt",
        "current_title": "",
        "current_firm": "",
        "location": "",
        "linkedin_url": "https://www.linkedin.com/in/averylinholt",
        "source_query_id": 1,
        "source_lane": "people_search",
        "verify_before_outreach": True,
        "data_sufficiency": "insufficient",
        "evidence": [
            {
                "url": "https://www.linkedin.com/in/averylinholt",
                "title": "Avery Holt",
                "published_date": "",
                "highlights": ["### Investor - [Benchmark](url) (Current)"],
                # Padded to the old cap with a firm blurb, exactly as a
                # live row is: the page is cut, not neatly ended.
                "text_excerpt": (
                    _RENDER_PAGE.split("## Education")[0]
                    + "Benchmark is a Venture Capital firm. " * 60
                )[:2000],
                "retrieved_at": "2026-09-03T00:00:00+00:00",
            }
        ],
    }
    assert "profile_render" not in stored
    assert len(stored["evidence"][0]["text_excerpt"]) == 2000

    candidate = ExaCandidate.from_dict(stored)

    assert candidate.profile_render is not None
    assert candidate.current_title == "Investor"
    assert candidate.current_firm == "Benchmark"
    assert candidate.location == "San Francisco Bay Area"
    # The band is recomputed: the record now knows the seat.
    assert candidate.data_sufficiency == "sufficient"
    # The dedupe key is NOT re-derived — rows elsewhere point at it.
    assert candidate.identity_key == "linkedin.com/in/averylinholt"

    summary = candidate.to_profile_summary()
    assert [(e.title, e.company, e.start, e.end) for e in summary.experiences] == [
        ("Investor", "Benchmark", "Jan 2025", "Present"),
        ("Member of Technical Staff", "Tensor Harbor", "Jan 2023", "Jan 2024"),
    ]
    assert f"{TENURE_LABEL}: 4 years and 11 months" in summary.about

    rendered = _profile_to_text(summary)
    assert "- Investor at Benchmark (Jan 2025-Present)" in rendered
    assert "About:\naveryholt.xyz" in rendered
    # Education DOES read "None listed" here, and honestly so: the old
    # 2000-char cap severed the section before it was ever stored. Only
    # a re-fetch under the raised cap can recover it — recovery reads
    # the page that was kept, it does not invent the page that was not.
    assert summary.education == []
    assert "Experience:\nNone listed" not in rendered

    # ``location`` is an independent field and stays fill-only: a stored
    # value is never overwritten by the page.
    kept = dict(stored, location="Austin, Texas")
    assert ExaCandidate.from_dict(kept).location == "Austin, Texas"

    # A record whose only page is a team page is NOT parsed as a profile.
    team_page = dict(
        stored,
        linkedin_url="",
        evidence=[
            {
                "url": "https://www.kilnventures.com/team",
                "title": "Team",
                "text_excerpt": "# Our Team\n\n## Experience\n\n### Partner - Kiln",
            }
        ],
    )
    harvested = ExaCandidate.from_dict(team_page)
    assert harvested.profile_render is None
    assert harvested.current_title == ""
    assert harvested.data_sufficiency == "insufficient"


def test_seat_fills_atomically_from_one_source() -> None:
    """On reload too, title and firm come from ONE source or neither.

    The version this replaced blessed the mix: a stored title of "Chief
    of Staff" kept, with "Benchmark" filled in beside it from the page.
    That composite seat exists in neither source and reads to a judge as
    a fact.
    """

    render = parse_exa_linkedin_markdown(_RENDER_PAGE)
    assert render is not None
    base = _candidate(current_title="", current_firm="").to_dict()
    base["evidence"] = [
        {
            "url": "https://www.linkedin.com/in/averylinholt",
            "title": "Avery Holt",
            "text_excerpt": _RENDER_PAGE,
        }
    ]
    base.pop("profile_render")

    # A stored title with no firm: BOTH are replaced by the page's seat.
    title_only = ExaCandidate.from_dict(dict(base, current_title="Chief of Staff"))
    assert (title_only.current_title, title_only.current_firm) == (
        "Investor",
        "Benchmark",
    )

    # A stored firm with no title: same rule, same direction.
    firm_only = ExaCandidate.from_dict(dict(base, current_firm="Northwind Capital"))
    assert (firm_only.current_title, firm_only.current_firm) == (
        "Investor",
        "Benchmark",
    )

    # Neither stored: the page's seat, as a pair.
    neither = ExaCandidate.from_dict(dict(base))
    assert (neither.current_title, neither.current_firm) == ("Investor", "Benchmark")
    assert neither.data_sufficiency == "sufficient"

    # A COMPLETE stored pair is already one source and stands untouched.
    complete = ExaCandidate.from_dict(
        dict(base, current_title="Chief of Staff", current_firm="Northwind Capital")
    )
    assert (complete.current_title, complete.current_firm) == (
        "Chief of Staff",
        "Northwind Capital",
    )
    # The render is still attached — the history reaches the judge even
    # when the stored seat wins.
    assert complete.profile_render is not None


def test_from_dict_rejects_a_render_whose_page_name_is_someone_else() -> None:
    """The evidence list is not proof the page is this person's."""

    base = _candidate(current_title="", current_firm="").to_dict()
    base["evidence"] = [
        {
            "url": "https://www.linkedin.com/in/averylinholt",
            "title": "Avery Holt",
            "text_excerpt": _RENDER_PAGE,  # the page says "Avery Holt"
        }
    ]
    base.pop("profile_render")

    stranger = ExaCandidate.from_dict(
        dict(base, name="Omar Vestergaard", data_sufficiency="insufficient")
    )
    assert stranger.profile_render is None
    assert stranger.render() is None
    assert stranger.current_title == "" and stranger.current_firm == ""
    # Nothing was recovered, so the stored band is left exactly as it was.
    assert stranger.data_sufficiency == "insufficient"
    # And nothing off the stranger's page reaches the profile summary.
    assert stranger.to_profile_summary().experiences == []

    # A middle initial is the same person and still binds.
    variant = ExaCandidate.from_dict(dict(base, name="Avery L. Holt"))
    assert variant.profile_render is not None
    assert variant.current_firm == "Benchmark"


def test_profile_summary_falls_back_to_seat_when_render_has_no_experiences() -> None:
    """An empty Experience list beside a known seat is a claim nobody made.

    A page cut before its Experience section, or a shape this parser
    does not know, yields a render with zero positions. Rendering that
    as "Experience: None listed" tells the judge this person has no
    career — which neither the page nor the record says.
    """

    seatless = parse_exa_linkedin_markdown(
        "# Avery Holt\n\nInvestor @ Benchmark\n\nSan Francisco Bay Area (US)\n\n"
        "## About\n\naveryholt.xyz\n"
    )
    assert seatless is not None and seatless.experiences == []

    candidate = _candidate(
        current_title="Investor",
        current_firm="Benchmark",
        location="San Francisco",
        profile_render=seatless.to_dict(),
    )

    summary = candidate.to_profile_summary()

    assert len(summary.experiences) == 1
    entry = summary.experiences[0]
    assert (entry.title, entry.company, entry.location) == (
        "Investor",
        "Benchmark",
        "San Francisco",
    )
    # A most-recent read it is not: no dates are invented for it.
    assert entry.start == "" and entry.end == ""
    # The About text the render DID carry still reaches the judge.
    assert "averyholt.xyz" in summary.about
    assert "Experience:\nNone listed" not in _profile_to_text(summary)

    # A position block with neither a title nor a company is not an
    # experience, and does not count against the fallback.
    blank = parse_exa_linkedin_markdown(_RENDER_PAGE)
    assert blank is not None
    for entry_ in blank.experiences:
        entry_.title, entry_.company = "", ""
    fallback = _candidate(
        current_title="Investor", current_firm="Benchmark",
        profile_render=blank.to_dict(),
    ).to_profile_summary()
    assert [(e.title, e.company) for e in fallback.experiences] == [
        ("Investor", "Benchmark")
    ]

    # And a record with neither a render seat nor a stored one honestly
    # has no experience to show.
    empty = _candidate(
        current_title="", current_firm="", profile_render=seatless.to_dict()
    )
    assert empty.to_profile_summary().experiences == []


def test_to_evidence_text_lists_every_evidence_url() -> None:
    candidate = _candidate()
    rendered = candidate.to_evidence_text()
    assert "═══ EVIDENCE ═══" in rendered
    assert candidate.evidence_urls() == [
        "https://www.linkedin.com/in/averylinholt",
        "https://averyholt.xyz",
    ]
    for url in candidate.evidence_urls():
        assert url in rendered


def test_to_evidence_text_renders_excerpt_when_no_highlights() -> None:
    """A retrieved page without highlights still has to reach the judge.

    Otherwise the entry arrives as a URL, a title, and a date — a
    citation with nothing behind it — and the high-bar posture drops
    someone for thin evidence the record actually holds.
    """

    candidate = _candidate(
        evidence=[
            ExaEvidence(
                url="https://www.kilnventures.com/team",
                title="Team — Kiln Ventures",
                highlights=[],
                text_excerpt="Theo   Marsh\nspent four years\tat Canvasly.",
            )
        ]
    )
    rendered = candidate.to_evidence_text()

    # Whitespace-collapsed, one line, under the entry it belongs to.
    assert "    Excerpt: Theo Marsh spent four years at Canvasly." in rendered

    # An entry WITH highlights renders those instead — no double-dipping.
    with_highlights = _candidate()
    assert "Excerpt:" not in with_highlights.to_evidence_text()


def test_to_evidence_text_carries_verify_flag_structurally() -> None:
    """The flag drives a block; it is not prose bolted on unconditionally."""

    assert "═══ VERIFY BEFORE OUTREACH ═══" in _candidate().to_evidence_text()
    assert (
        "═══ VERIFY BEFORE OUTREACH ═══"
        not in _candidate(verify_before_outreach=False).to_evidence_text()
    )


# ---------------------------------------------------------------------------
# ExaSearchQuery / ExaProgress
# ---------------------------------------------------------------------------


def test_exa_search_query_round_trips_and_ignores_unknown_keys() -> None:
    query = ExaSearchQuery(
        id=1,
        name="people_search · Benchmark",
        lane="people_search",
        firm="Benchmark",
        query="Senior Associate or Principal at Benchmark SF Bay Area",
        category="people",
        include_domains=["linkedin.com"],
        exclude_terms=["Benchmark Capital Partners VII"],
    )
    payload = query.to_dict()
    payload["retired_field"] = "ignored"
    assert ExaSearchQuery.from_dict(payload) == query


def test_exa_progress_round_trips_through_file(tmp_path) -> None:
    progress = ExaProgress(
        brief_name="cos-v3",
        queries=[
            ExaSearchQuery(
                id=1,
                name="people_search · Benchmark",
                lane="people_search",
                firm="Benchmark",
                query="Senior Associate at Benchmark SF Bay Area",
                category="people",
                include_domains=["linkedin.com"],
            )
        ],
        candidates_discovered=53,
        candidates_saved=41,
        current_query_id=1,
        discovered_identity_keys=["linkedin.com/in/averylinholt"],
        api_calls_made=4,
        exa_spend_usd=1.25,
    )
    path = tmp_path / "state" / "exa_progress.json"
    progress.save(str(path))

    restored = ExaProgress.from_file(str(path))
    assert restored == progress
    assert restored.queries[0].lane == "people_search"
    assert restored.discovered_identity_keys == ["linkedin.com/in/averylinholt"]
    assert restored.exa_spend_usd == 1.25
