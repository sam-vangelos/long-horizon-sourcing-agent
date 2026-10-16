"""Exa source module S3 — page extraction and grounding coverage.

Offline: the deterministic parser needs nothing, and the model path
runs against fake ``llm`` callables. No network, no real cheap-tier
call.

The properties under test are anti-fabrication ones. A model asked to
read a firm's team page can invent a person, and it can attach a real
person to an invented title — the 2026-08-28 CoS slate hit both. Most
of what follows asks one question in different words: does the record
carry only what the retrieved page said, about the person it said it
about?
"""

from __future__ import annotations

import pytest

from exa.extraction import (
    RENDER_NAME_MISMATCH_NOTE,
    ParsedProfile,
    extract_people_from_page,
    ground_profile,
    parse_linkedin_result,
    quote_is_grounded,
)
from exa.schemas import (
    EVIDENCE_HIGHLIGHT_MAX_CHARS,
    EVIDENCE_HIGHLIGHT_MAX_COUNT,
    ExaEvidence,
)
from shared.failures import ApiBudgetExhaustedError


class _Result:
    """The subset of ``ExaResult`` the parser reads."""

    def __init__(self, url: str, title: str, text: str = "", highlights=None):
        self.url = url
        self.title = title
        self.text = text
        self.highlights = list(highlights or [])


# A realistic team page: three people, entries far enough apart that one
# person's title is outside the next person's grounding window. The
# window rule is meaningless against a two-sentence toy page.
_PAGE_TEXT = (
    "Cedarline Capital\nTeam\n\n"
    "Talia Brennan is Chief of Staff at Cedarline Capital, where she runs "
    "the firm's operating cadence, LP reporting, and platform hiring. She "
    "joined in 2024 from Halden Growth, where she led business operations.\n\n"
    "Omar Vestergaard, Senior Associate, invests in seed and Series A "
    "infrastructure software. Before Cedarline he was a founding engineer "
    "at Palmetto Loop.\n\n"
    "Rosalind Achebe is a General Partner and co-founded the firm in 2016.\n\n"
    "San Mateo, California"
)

_TALIA_QUOTE = "Talia Brennan is Chief of Staff at Cedarline Capital"
_OMAR_QUOTE = "Omar Vestergaard, Senior Associate"


# Exa's LinkedIn profile rendering, live shape with a synthetic name.
# The page title for a hit like this is the bare name, which is what made
# the title-only parse blank out title, firm, and location.
_LINKEDIN_RENDER = """# Talia Brennan

Benchmark

San Francisco Bay Area (US)

500 connections • 2,111 followers

## About

taliabrennan.xyz

Total Experience: 4 years and 11 months

## Experience

### Investor - [Benchmark](https://www.linkedin.com/company/benchmark-vc) (Current)

Jan 2025 - Present (1 year and 7 months) in San Francisco, California, United States

Department: Consulting

Benchmark is a Venture Capital and Private Equity Principals company.

### Research - [Helix Institute](https://www.linkedin.com/company/helix-institute-org)

Jan 2025 - Jan 2025 in Stanford, California, United States

Department: Research • Level: Specialist
"""


def _evidence(text: str = _PAGE_TEXT, highlights=None) -> ExaEvidence:
    return ExaEvidence(
        url="https://www.cedarlinecapital.com/team",
        title="Our Team | Cedarline Capital",
        highlights=list(highlights or []),
        text_excerpt=text,
    )


# ---------------------------------------------------------------------------
# parse_linkedin_result
# ---------------------------------------------------------------------------


def test_parse_linkedin_result_name_title_firm() -> None:
    parsed = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title=(
                "Dana Okafor - Chief of Staff to the CEO - Northwind Capital "
                "| LinkedIn"
            ),
        )
    )

    assert parsed is not None
    assert parsed.name == "Dana Okafor"
    assert parsed.title == "Chief of Staff to the CEO"
    assert parsed.firm == "Northwind Capital"
    assert parsed.linkedin_url == (
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff"
    )


def test_parse_linkedin_result_name_only_title() -> None:
    """The two-segment shapes that resolve, and the one-segment shape."""

    title_only = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="Dana Okafor - Chief of Staff to the CEO | LinkedIn",
        )
    )
    assert title_only is not None
    assert title_only.title == "Chief of Staff to the CEO"
    assert title_only.firm == ""

    firm_only = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="Dana Okafor - Northwind Capital | LinkedIn",
        )
    )
    assert firm_only is not None
    assert firm_only.title == ""
    assert firm_only.firm == "Northwind Capital"

    bare = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="Dana Okafor | LinkedIn",
        )
    )
    assert bare is not None
    assert bare.name == "Dana Okafor"
    assert bare.title == ""
    assert bare.firm == ""


def test_parse_two_segment_ambiguous_leaves_both_blank() -> None:
    """When the segment could be either, it becomes neither.

    The page title still reaches the judge through ``ExaEvidence.title``,
    so refusing to classify costs nothing — while a guess written into
    ``current_firm`` becomes a fact the judge reasons from.
    """

    ambiguous = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/jane-doe",
            title="Jane Doe - Investment Professional | LinkedIn",
        )
    )
    assert ambiguous is not None
    assert ambiguous.name == "Jane Doe"
    assert ambiguous.title == ""
    assert ambiguous.firm == ""

    # A firm suffix with no title word resolves to a firm, even though
    # "Lead" opens the name.
    firm = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/jane-doe",
            title="Jane Doe - Lead Edge Capital | LinkedIn",
        )
    )
    assert firm is not None
    assert firm.firm == "Lead Edge Capital"
    assert firm.title == ""

    # Both signals at once is still ambiguous: "Partner" is a title and
    # "Partners" ends a firm name.
    both = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/jane-doe",
            title="Jane Doe - Principal at Wexford Peak Partners | LinkedIn",
        )
    )
    assert both is not None
    assert (both.title, both.firm) == ("", "")


def test_parse_linkedin_result_en_dash_separator() -> None:
    en_dash = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/marcus-lindqvist",
            title="Marcus Lindqvist – Chief of Staff – Riverstone Partners | LinkedIn",
        )
    )
    assert en_dash is not None
    assert (en_dash.name, en_dash.title, en_dash.firm) == (
        "Marcus Lindqvist",
        "Chief of Staff",
        "Riverstone Partners",
    )

    em_dash = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/marcus-lindqvist",
            title="Marcus Lindqvist — Chief of Staff — Riverstone Partners | LinkedIn",
        )
    )
    assert em_dash is not None
    assert em_dash.firm == "Riverstone Partners"

    # A hyphenated given name is not a separator: only a spaced dash is.
    hyphenated = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/jean-luc-arnaud",
            title="Jean-Luc Arnaud - Principal - Alder Peak | LinkedIn",
        )
    )
    assert hyphenated is not None
    assert hyphenated.name == "Jean-Luc Arnaud"
    assert hyphenated.firm == "Alder Peak"


def test_parse_linkedin_result_rejects_non_profile_url() -> None:
    company = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/company/northwind-capital",
            title="Northwind Capital - Venture Capital | LinkedIn",
        )
    )
    assert company is None

    article = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/pulse/what-a-chief-of-staff-does-okafor",
            title="Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
        )
    )
    assert article is None

    nameless = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="| LinkedIn",
        )
    )
    assert nameless is None


def test_parse_linkedin_result_location_only_when_shaped() -> None:
    shaped = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
            text=(
                "Dana Okafor\n"
                "San Francisco, California\n"
                "Chief of Staff to the CEO at Northwind Capital."
            ),
        )
    )
    assert shaped is not None
    assert shaped.location == "San Francisco, California"

    prose = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
            text=(
                "Chief of Staff to the CEO at Northwind Capital. Partner-track "
                "operator focused on portfolio operations. San Francisco Bay Area."
            ),
        )
    )
    assert prose is not None
    assert prose.location == ""

    buried = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
            text="one\ntwo\nthree\nfour\nfive\nsix\nAustin, Texas",
        )
    )
    assert buried is not None
    assert buried.location == ""


def test_parse_linkedin_result_recovers_title_firm_from_markdown_when_title_is_bare_name() -> None:
    """The 2026-09-03 failure, in one test.

    Exa's title for a people-search hit is the bare name, so the segment
    rules find no title and no firm. The page TEXT carries both, as
    structured markdown — and until S7 nobody read it, so every in-band
    candidate went to the judge with ``data_sufficiency=insufficient``.
    """

    parsed = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan",  # no separators — the bare name
            text=_LINKEDIN_RENDER,
        )
    )

    assert parsed is not None
    assert parsed.name == "Talia Brennan"
    assert parsed.title == "Investor"
    assert parsed.firm == "Benchmark"
    assert parsed.location == "San Francisco Bay Area"

    # The whole page rides along, so the record can carry the history.
    assert parsed.render is not None
    assert [e.company for e in parsed.render.experiences] == [
        "Benchmark",
        "Helix Institute",
    ]

    # Everything recovered IS on the retrieved page, so the substring
    # grounding the record's anti-fabrication guard rests on passes by
    # construction. (The proximity rule in ``ground_profile`` is a
    # different check, and the people_search lane does not run it —
    # ``exa/acquisition.py:473``.)
    evidence = ExaEvidence(
        url="https://www.linkedin.com/in/talia-brennan",
        title="Talia Brennan",
        text_excerpt=_LINKEDIN_RENDER,
    )
    for value in (parsed.name, parsed.title, parsed.firm, parsed.location):
        assert quote_is_grounded(value, evidence)

    # A page title that DOES carry a title and firm still wins: two
    # readings of one page, and the first is not overwritten.
    titled = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan - Chief of Staff - Northwind Capital | LinkedIn",
            text=_LINKEDIN_RENDER,
        )
    )
    assert titled is not None
    assert titled.title == "Chief of Staff"
    assert titled.firm == "Northwind Capital"
    assert titled.render is not None


def test_bay_area_location_line_is_accepted() -> None:
    """"San Francisco Bay Area (US)" is a location; the old scan rejected it.

    It has no comma, so ``_extract_location``'s "City, Region" shape
    never matched — which is why 40 of 40 live records carried an empty
    location while their pages stated one.
    """

    bay_area = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan",
            text=_LINKEDIN_RENDER,
        )
    )
    assert bay_area is not None
    assert bay_area.location == "San Francisco Bay Area"

    # The "City, State, Country (US)" shape loses only the country code.
    city_state = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan",
            text=_LINKEDIN_RENDER.replace(
                "San Francisco Bay Area (US)",
                "Menlo Park, California, United States (US)",
            ),
        )
    )
    assert city_state is not None
    assert city_state.location == "Menlo Park, California, United States"

    # A headline that merely ends in a country code is not a location.
    headline_only = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan",
            text=_LINKEDIN_RENDER.replace(
                "San Francisco Bay Area (US)", "Crypto Strategy at Visa (US)"
            ),
        )
    )
    assert headline_only is not None
    assert headline_only.location == ""

    # A page that states no location still states none.
    no_location = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan",
            text="# Talia Brennan\n\nInvestor at Benchmark\n\n## Experience\n\n"
            "### Investor - Benchmark (Current)\n\nJan 2025 - Present\n",
        )
    )
    assert no_location is not None
    assert no_location.location == ""


def test_render_is_rejected_when_page_name_differs() -> None:
    """A LinkedIn render must be THIS person's page, or it is not attached.

    A record's evidence is not guaranteed to hold its own subject — a
    merge, a mis-attributed URL, a redirect. Grafting a stranger's whole
    career onto a candidate is worse than the thin profile S7 fixed: a
    thin record understates, a bound-wrong record fabricates.
    """

    stranger = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/omar-vestergaard",
            title="Omar Vestergaard",
            # The retrieved page is Talia Brennan's.
            text=_LINKEDIN_RENDER,
        )
    )

    assert stranger is not None
    assert stranger.name == "Omar Vestergaard"
    assert stranger.render is None
    assert stranger.render_note == RENDER_NAME_MISMATCH_NOTE
    # Nothing off the page reaches the record — not the seat, not the
    # location.
    assert stranger.title == "" and stranger.firm == ""
    assert stranger.location == ""

    # A middle initial, a comma-reversed name, or a diacritic is the
    # same person and still binds.
    for variant in ("Talia S. Brennan", "Brennan, Talia", "Tália Brennan"):
        matched = parse_linkedin_result(
            _Result(
                url="https://www.linkedin.com/in/talia-brennan",
                title=variant,
                text=_LINKEDIN_RENDER,
            )
        )
        assert matched is not None, variant
        assert matched.render is not None, variant
        assert matched.render_note == "", variant
        assert (matched.title, matched.firm) == ("Investor", "Benchmark")

    # A page that is not a render at all carries no note — there was
    # nothing to reject.
    not_a_render = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/omar-vestergaard",
            title="Omar Vestergaard",
            text="Omar Vestergaard\nSan Francisco, California\nSenior Associate.",
        )
    )
    assert not_a_render is not None
    assert not_a_render.render is None and not_a_render.render_note == ""


def test_seat_fills_atomically_from_one_source() -> None:
    """Title and firm come from ONE source or neither is touched.

    "Chief of Staff at Benchmark" — a title off the page title, a firm
    off the render — is a seat that appears in neither source. It reads
    as a fact and is a composition.
    """

    # Title-only from the page title: the render's seat replaces BOTH.
    title_only = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan - Chief of Staff | LinkedIn",
            text=_LINKEDIN_RENDER,
        )
    )
    assert title_only is not None
    assert (title_only.title, title_only.firm) == ("Investor", "Benchmark")
    assert title_only.firm != "" and title_only.title != "Chief of Staff"

    # Firm-only from the page title: same rule, same direction.
    firm_only = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan - Northwind Capital | LinkedIn",
            text=_LINKEDIN_RENDER,
        )
    )
    assert firm_only is not None
    assert (firm_only.title, firm_only.firm) == ("Investor", "Benchmark")

    # A COMPLETE pair from the page title stands — one source already.
    complete = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan - Chief of Staff - Northwind Capital | LinkedIn",
            text=_LINKEDIN_RENDER,
        )
    )
    assert complete is not None
    assert (complete.title, complete.firm) == ("Chief of Staff", "Northwind Capital")

    # No render to take a pair from: the partial survives as it was.
    no_render = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/talia-brennan",
            title="Talia Brennan - Chief of Staff | LinkedIn",
            text="Talia Brennan\nSan Francisco, California\nChief of Staff.",
        )
    )
    assert no_render is not None
    assert (no_render.title, no_render.firm) == ("Chief of Staff", "")


def test_parse_linkedin_result_ignores_highlights_for_location() -> None:
    """A highlight's position says nothing about whose location it is.

    Exa picks highlights by relevance, so a shaped fragment lifted out
    of the middle of a roster would be attributed to whoever happened to
    be parsed.
    """

    parsed = parse_linkedin_result(
        _Result(
            url="https://www.linkedin.com/in/dana-okafor-chiefofstaff",
            title="Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
            text="Chief of Staff to the CEO at Northwind Capital.",
            highlights=["Boston, Massachusetts"],
        )
    )

    assert parsed is not None
    assert parsed.location == ""


# ---------------------------------------------------------------------------
# ground_profile
# ---------------------------------------------------------------------------


def test_ground_profile_drops_name_not_on_page() -> None:
    ghost = ParsedProfile(
        name="Ghost Person",
        title="Managing Director",
        firm="Cedarline Capital",
        # A genuine sentence from the page, attached to a person who is
        # not in it. The quote check alone would let this through.
        evidence_quote=_TALIA_QUOTE,
    )

    assert ground_profile(ghost, _evidence()) is None

    # And a real person whose quote does not name them is equally a
    # mismatch: the quote proves a sentence, not whose sentence it is.
    mismatched = ParsedProfile(
        name="Omar Vestergaard", title="", firm="", evidence_quote=_TALIA_QUOTE
    )
    assert ground_profile(mismatched, _evidence()) is None

    # Both name tokens are on this page — "Halden" in Talia's history,
    # "Brennan" in her name — but 150 characters apart, so they are not
    # one person being named. A page that merely contains the words is
    # not a page that names the human.
    scattered = ParsedProfile(
        name="Halden Brennan", title="", firm="", evidence_quote=_TALIA_QUOTE
    )
    assert ground_profile(scattered, _evidence()) is None


def test_ground_profile_blanks_title_outside_name_window() -> None:
    """Talia's title cannot travel down the page to Omar."""

    person = ParsedProfile(
        name="Omar Vestergaard",
        title="Chief of Staff",
        firm="Cedarline Capital",
        evidence_quote=_OMAR_QUOTE,
    )

    grounded = ground_profile(person, _evidence())

    assert grounded is not None
    assert grounded.name == "Omar Vestergaard"
    assert grounded.title == ""
    # Talia's firm mention is in her paragraph, not his.
    assert grounded.firm == ""

    # His own title, in his own sentence, survives.
    his_own = ground_profile(
        ParsedProfile(
            name="Omar Vestergaard",
            title="Senior Associate",
            evidence_quote=_OMAR_QUOTE,
        ),
        _evidence(),
    )
    assert his_own is not None
    assert his_own.title == "Senior Associate"

    # And Talia keeps hers.
    hers = ground_profile(
        ParsedProfile(
            name="Talia Brennan",
            title="Chief of Staff",
            firm="Cedarline Capital",
            evidence_quote=_TALIA_QUOTE,
        ),
        _evidence(),
    )
    assert hers is not None
    assert (hers.title, hers.firm) == ("Chief of Staff", "Cedarline Capital")


def test_ground_profile_does_not_span_highlight_boundaries() -> None:
    """Distance only means something inside one fragment.

    Highlights are lifted from different places on the page, so a corpus
    that joined them end to end would put Talia's title within a few
    characters of Omar's name and hand him her seat.
    """

    evidence = _evidence(
        text=_OMAR_QUOTE + ", invests in seed infrastructure software.",
        highlights=[_TALIA_QUOTE + ", where she runs the operating cadence."],
    )

    grounded = ground_profile(
        ParsedProfile(
            name="Omar Vestergaard",
            title="Chief of Staff",
            firm="Cedarline Capital",
            evidence_quote=_OMAR_QUOTE,
        ),
        evidence,
    )

    assert grounded is not None
    assert (grounded.title, grounded.firm) == ("", "")


def test_ground_profile_token_boundary_partner_vs_partnerships() -> None:
    evidence = _evidence(
        "Ingrid Solheim leads Partnerships at Wexford Peak Partners."
    )
    quote = "Ingrid Solheim leads Partnerships"

    inflated = ground_profile(
        ParsedProfile(name="Ingrid Solheim", title="Partner", evidence_quote=quote),
        evidence,
    )
    assert inflated is not None
    assert inflated.title == ""

    actual = ground_profile(
        ParsedProfile(
            name="Ingrid Solheim", title="Partnerships", evidence_quote=quote
        ),
        evidence,
    )
    assert actual is not None
    assert actual.title == "Partnerships"


def test_ground_profile_drops_invented_linkedin_url() -> None:
    """An invented URL is worse than none: it mints a strong identity key."""

    invented = ground_profile(
        ParsedProfile(
            name="Talia Brennan",
            linkedin_url="https://www.linkedin.com/in/talia-brennan",
            evidence_quote=_TALIA_QUOTE,
        ),
        _evidence(),
    )
    assert invented is not None
    assert invented.linkedin_url == ""

    # A URL whose slug is actually printed on the page survives.
    evidence = _evidence(
        _PAGE_TEXT
        + "\n\nIngrid Solheim: https://www.linkedin.com/in/ingrid-solheim-wexford"
    )
    real = ground_profile(
        ParsedProfile(
            name="Ingrid Solheim",
            linkedin_url="https://linkedin.com/in/ingrid-solheim-wexford",
            evidence_quote="Ingrid Solheim: https://www.linkedin.com/in/ingrid-solheim-wexford",
        ),
        evidence,
    )
    assert real is not None
    assert real.linkedin_url == "https://linkedin.com/in/ingrid-solheim-wexford"


def test_ground_profile_matches_across_diacritics_and_middle_names() -> None:
    evidence = _evidence("Ávery Lin Holt is an Investor at Benchmark.")

    grounded = ground_profile(
        ParsedProfile(
            name="Avery Holt",
            title="Investor",
            firm="Benchmark",
            evidence_quote="Avery Lin Holt is an Investor at Benchmark",
        ),
        evidence,
    )

    assert grounded is not None
    assert (grounded.title, grounded.firm) == ("Investor", "Benchmark")


# ---------------------------------------------------------------------------
# extract_people_from_page
# ---------------------------------------------------------------------------


def test_extract_people_uses_fake_llm_and_validates_quotes() -> None:
    captured: dict = {}

    def _llm(system_prompt, user_prompt, expect_json=True, usage_context=None):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        captured["usage_context"] = usage_context
        return {
            "people": [
                {
                    "name": "Talia Brennan",
                    "title": "Chief of Staff",
                    "firm": "Cedarline Capital",
                    "location": "",
                    "linkedin_url": "",
                    "evidence_quote": _TALIA_QUOTE,
                },
                {
                    "name": "Priya Raghunathan",
                    "title": "Managing Director",
                    "firm": "Cedarline Capital",
                    "location": "",
                    "linkedin_url": "",
                    "evidence_quote": (
                        "Priya Raghunathan is Managing Director at Cedarline "
                        "Capital"
                    ),
                },
            ]
        }

    people = extract_people_from_page(
        _evidence(),
        firm_hint="Cedarline Capital",
        llm=_llm,
        usage_context={"lane_id": "team_page", "string_id": 7},
    )

    # The fabricated person's quote is nowhere on the page, so she is not
    # a person this run ever heard of.
    assert [person.name for person in people] == ["Talia Brennan"]
    assert people[0].title == "Chief of Staff"
    assert people[0].firm == "Cedarline Capital"

    assert captured["usage_context"]["stage"] == "exa_extraction"
    assert captured["usage_context"]["source"] == "exa"
    assert captured["usage_context"]["lane_id"] == "team_page"
    # The prompt is built from the same text the validator checks
    # against; otherwise the substring gate is a coin flip.
    assert "Talia Brennan is Chief of Staff" in captured["user"]


def test_extraction_context_canonical_keys_win() -> None:
    captured: dict = {}

    def _llm(system_prompt, user_prompt, expect_json=True, usage_context=None):
        captured.update(usage_context or {})
        return {"people": []}

    extract_people_from_page(
        _evidence(),
        firm_hint="Cedarline Capital",
        llm=_llm,
        usage_context={
            "source": "linkedin",
            "stage": "facial_judgment",
            "module": "linkedin",
            "run_id": "run-7",
        },
    )

    assert captured["source"] == "exa"
    assert captured["stage"] == "exa_extraction"
    assert captured["module"] == "exa"
    # Keys the caller owns still ride along.
    assert captured["run_id"] == "run-7"


def test_extract_people_fail_soft_on_bad_json(capsys) -> None:
    def _llm(system_prompt, user_prompt, expect_json=True, usage_context=None):
        return "not json at all, sorry"

    people = extract_people_from_page(
        _evidence(), firm_hint="Cedarline Capital", llm=_llm
    )

    assert people == []
    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(printed) == 1
    assert printed[0].startswith(
        "    [exa] extraction failed for https://www.cedarlinecapital.com/team:"
    )

    def _boom(system_prompt, user_prompt, expect_json=True, usage_context=None):
        raise RuntimeError("cheap tier unavailable")

    assert (
        extract_people_from_page(
            _evidence(), firm_hint="Cedarline Capital", llm=_boom
        )
        == []
    )
    boom_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    assert len(boom_lines) == 1
    assert "cheap tier unavailable" in boom_lines[0]


def test_extract_people_propagates_budget_exhaustion() -> None:
    """Spend exhaustion is the RUN's problem, not this page's.

    Absorbed, it would look exactly like a page that named nobody — and
    the campaign would work through its whole query list producing an
    empty slate while the console reported healthy extractions.
    """

    def _llm(system_prompt, user_prompt, expect_json=True, usage_context=None):
        raise ApiBudgetExhaustedError("LLM spend cap reached")

    with pytest.raises(ApiBudgetExhaustedError):
        extract_people_from_page(
            _evidence(), firm_hint="Cedarline Capital", llm=_llm
        )


def test_extract_people_never_infers_title_when_absent() -> None:
    """A title the page does not carry beside the person is blanked.

    The quote check alone does NOT catch this class: a model can quote a
    real sentence and still report a title that sentence never says.
    """

    def _llm(system_prompt, user_prompt, expect_json=True, usage_context=None):
        return {
            "people": [
                {
                    "name": "Omar Vestergaard",
                    "title": "Managing Partner",
                    "firm": "Cedarline Capital",
                    "location": "",
                    "linkedin_url": "",
                    "evidence_quote": _OMAR_QUOTE,
                }
            ]
        }

    people = extract_people_from_page(
        _evidence(), firm_hint="Cedarline Capital", llm=_llm
    )

    assert len(people) == 1
    assert people[0].name == "Omar Vestergaard"
    assert people[0].title == ""

    def _capture(system_prompt, user_prompt, expect_json=True, usage_context=None):
        assert "Never infer" in system_prompt
        assert "exact title text you reported" in system_prompt
        return {"people": []}

    assert (
        extract_people_from_page(
            _evidence(), firm_hint="Cedarline Capital", llm=_capture
        )
        == []
    )


def test_quote_grounding_ignores_whitespace_and_case_but_not_emptiness() -> None:
    evidence = _evidence()

    assert quote_is_grounded("TALIA   BRENNAN\nis Chief of Staff", evidence)
    assert not quote_is_grounded("", evidence)
    assert not quote_is_grounded("   ", evidence)
    assert not quote_is_grounded("Talia Brennan is Managing Partner", evidence)


# ---------------------------------------------------------------------------
# Evidence ingestion caps
# ---------------------------------------------------------------------------


def test_exa_evidence_caps_highlights_at_ingestion() -> None:
    """Count first, then length — and the object matches its own dict.

    ``to_evidence_text`` renders every highlight into the evaluation
    prompt with no cap of its own (`exa/schemas.py:285-286`), so an
    unusually generous response would inflate both the prompt and the
    progress file.
    """

    evidence = ExaEvidence(
        url="https://example.com/team",
        highlights=[f"highlight {index} " + "x" * 900 for index in range(25)],
    )

    assert len(evidence.highlights) == EVIDENCE_HIGHLIGHT_MAX_COUNT
    assert all(
        len(highlight) <= EVIDENCE_HIGHLIGHT_MAX_CHARS
        for highlight in evidence.highlights
    )
    assert evidence.highlights == evidence.to_dict()["highlights"]
    # Round trip is stable rather than shrinking on every save/load.
    assert ExaEvidence.from_dict(evidence.to_dict()).highlights == evidence.highlights
