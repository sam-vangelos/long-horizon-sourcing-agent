"""Exa source module S7 — the LinkedIn markdown render parser.

Exa returns a LinkedIn profile as structured markdown, and until S7
nothing read it: ``parse_linkedin_result`` read the page title, which for
a people-search hit is the bare name. Every in-band candidate of the
2026-09-03 CoS run therefore reached the judge with no title, no firm, no
About, and one dateless experience — and 100% of them came back
``PARSE_FAILURE``.

The fixtures below are the live page shape with synthetic names. Both
were built from redacted samples of that run, so what is asserted here is
the rendering Exa actually produces, not a shape invented for the test.

Offline by construction: the parser is deterministic and takes a string.
"""

from __future__ import annotations

from exa.linkedin_render import (
    BULLET_MAX_CHARS,
    BULLETS_PER_POSITION_MAX,
    LinkedInRender,
    RenderedEducation,
    RenderedExperience,
    bind_render,
    duration_to_years,
    looks_like_location,
    parse_exa_linkedin_markdown,
    render_names_match,
    seat_pair,
    strip_country_suffix,
)


# The common shape: a headline line, a location line, an About section
# that carries the personal site and the Total Experience line, and
# position blocks each followed by Exa's own company blurb.
INVESTOR_PAGE = """# Talia Brennan

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

Benchmark is a Venture Capital and Private Equity Principals company. Benchmark has 40-50 employees (+10% YoY) and has annual revenue in the $10M-$20M range, founded in 1995. Headquartered in woodside, california, United States, with offices in San Francisco and San Mateo.

### Research - [Helix Institute](https://www.linkedin.com/company/helix-institute-org)

Jan 2025 - Jan 2025 in Stanford, California, United States

Department: Research • Level: Specialist

Helix Institute is a nonprofit organization. Helix Institute has 200-300 employees (+40% YoY), founded in 2021. Headquartered in Palo Alto, California, United States.

### Member of Technical Staff - [Tensor Harbor](https://www.linkedin.com/company/tensor-harbor)

Jan 2023 - Jan 2024 (1 year) in New York, United States

Department: Engineering and Technical
"""


# The second shape: two positions grouped under ONE company header, a
# position with no company link at all, and an education section. Built
# from the second redacted sample plus the education block observed on
# four of the run's forty pages.
OPERATOR_PAGE = """# Omar Vestergaard

Investor @ Kiln Ventures

San Francisco Bay Area (US)

500 connections • 1,795 followers

## About

Investor @ Kiln Ventures

Total Experience: 14 years and 8 months

## Experience

### Principal - [Kiln Ventures](https://www.linkedin.com/company/kiln-ventures) (Current)

Feb 2022 - Present (4 years and 6 months)

Department: Education • Level: Senior

Kiln Ventures is a Financial Services company. Kiln Ventures has 30-40 employees, founded in 2017.

### Advisor & Angel Investor - Various Startups (Current)

Jan 2020 - Present (5 years and 2 months)

Department: Other

### [Canvasly](https://www.linkedin.com/company/canvasly)

#### Self-Serve Strategy

Dec 2020 - Feb 2022 (1 year and 2 months)

Department: Consulting • Level: Specialist

#### Strategic Finance & Business Operations

Jan 2018 - Dec 2020 (2 years and 11 months)

Department: Finance & Accounting • Level: Specialist

Canvasly (CNVS) is a Design Services company. Canvasly has 2,000-3,000 employees (+20% YoY), founded in 2012. IPO 2025.

## Education

### M.B.A at [Stanford Graduate School of Business](https://www.linkedin.com/school/stanford-gsb)

1982 - 1984 (2 years)

### BA at Rice University

1975 - 1979 (4 years)

## Honors & Awards

### Founders Prize for Young Leadership
"""


# A third live shape: no headline at all, and no About section — the
# location sits directly under the name. Eight of the run's forty pages
# looked like this, which is why the header is read by shape and not by
# line position.
HEADLINE_LESS_PAGE = """# Rosalind Achebe

Atherton, California, United States (US)

236 connections • 1,637 followers

## Experience

### Co-Host - PlayLoop Podcast (Current)

Nov 2022 - Present (3 years and 9 months) in https://www.playlooppod.com/
"""


# A fourth live shape: the person's own position bullets. They sit between a
# position's date line and its Department/Level line, and a company group
# can hold one position with bullets and one without. Built from a stored
# 2026-09-23 probe page with a synthetic name; the firm blurbs are Exa's.
BULLETED_PAGE = """# Jordan Okafor

AI or something

Fort Collins, Colorado, United States (US)

216 connections • 224 followers

## About

Python, data science, and frontier LLM evaluation.

Total Experience: 5 years and 7 months

## Experience

### Data Operations - [Surge AI](https://www.linkedin.com/company/surge-ai) (Current)

Jan 2025 - Present (1 year and 8 months) in Remote

Department: Operations • Level: Mid

Surge AI is a Software Development company. Surge AI has 200-300 employees (+150% YoY).

### [SpaceXAI](https://www.linkedin.com/company/spacexai)

#### Human Data Expert Team Lead

Dec 2024 - Sep 2025 (9 months)

- Wrote and conducted technical interviews to grow a team of Software Engineers.
- Led a pod (10-15 people) in solving technical issues related to improving LLM programming ability.
- Maintained internal tooling and coordinated contributions from direct reports.

Department: Research • Level: Head

#### AI Coding Tutor

Oct 2024 - Dec 2024 (2 months)

Department: Education • Level: Specialist

xAI is a Technology, Information and Internet company. xAI has 2,000-3,000 employees (+90% YoY).

- A stray dash line after the firm blurb belongs to nobody.

### Undergraduate Research Assistant - Hollis Lab, Front Range State University

Oct 2021 - May 2024 (2 years and 7 months) in Fort Collins, Colorado, United States

- Wrote Java and Python software to optimize workflows and allow for analysis of high-throughput imaging.
-    Trained new hires on   basic laboratory techniques.
- Department: Biology. Built the lab's imaging pipeline.

Department: Research • Level: Entry
"""


# A fifth live shape: a position described in PROSE rather than bullets, the
# most common shape in the 2026-09-23 probe corpus (3,049 prose-only blocks
# against 745 bullet-only). The Queue Manager block is a stored page's text
# with a synthetic name; the firm blurbs are Exa's. The second block mixes
# prose and bullets; the third has no Department/Level line at all, and its
# prose is followed by a blurb; the fourth has prose and nothing after it.
PROSE_PAGE = """# Casey Lindqvist

Queue Manager at Scale AI

Columbus, Ohio, United States (US)

300 connections • 310 followers

## Experience

### Queue Manager (HireArt contractor) - [Scale AI](https://www.linkedin.com/company/scaleai) (Current)

Sep 2025 - Present (1 year)

As a Queue Manager, I ensure seamless project execution by managing task workflows, facilitating communication, and maintaining high-quality standards across multiple teams. I proactively resolve issues impacting task delivery, lead contributor sessions for real-time support, and oversee quality assurance through audits and feedback.

Department: Operations • Level: Manager

Scale AI is a Software Development company. Scale AI delivers high quality training data for AI applications. Scale AI has 3,000-4,000 employees (+30% YoY), founded in 2016.

### Delivery Lead - [Mercor](https://www.linkedin.com/company/mercor)

Jan 2024 - Aug 2025 (1 year and 7 months)

Ran the coding-evaluation program for two frontier labs.

- Hired and managed 40 contract reviewers.
- Built the rubric audit loop.

Department: Operations • Level: Lead

Mercor is a Technology, Information and Internet company. Mercor has 500-600 employees.

### Analyst - Halden Group

Jun 2022 - Dec 2023 (1 year and 6 months)

Wrote the weekly pipeline-quality report read by the COO.

Halden Group is a Business Consulting and Services company. Halden Group has 50-100 employees, founded in 2001.

- A stray dash line after this blurb belongs to nobody.

### Research Assistant - Hollis Lab

Sep 2020 - May 2022 (1 year and 8 months)

Maintained the imaging pipeline and trained new hires.
"""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def test_parses_name_headline_location_about() -> None:
    render = parse_exa_linkedin_markdown(INVESTOR_PAGE)

    assert render is not None
    assert render.name == "Talia Brennan"
    assert render.headline == "Benchmark"
    assert render.location == "San Francisco Bay Area (US)"
    # The About prose is the person's own words; the computed
    # Total Experience line is lifted out of it.
    assert render.about == "taliabrennan.xyz"
    assert "Total Experience" not in render.about
    # The social-proof line is page chrome, never a headline.
    assert "connections" not in render.headline
    assert "followers" not in render.headline

    # The "City, State, Country" location shape parses too.
    other = parse_exa_linkedin_markdown(OPERATOR_PAGE)
    assert other is not None
    assert other.headline == "Investor @ Kiln Ventures"
    assert other.about == "Investor @ Kiln Ventures"

    # No headline line: the location must not slide into the headline.
    bare = parse_exa_linkedin_markdown(HEADLINE_LESS_PAGE)
    assert bare is not None
    assert bare.name == "Rosalind Achebe"
    assert bare.headline == ""
    assert bare.location == "Atherton, California, United States (US)"
    assert bare.about == ""


def test_duplicate_section_heading_is_ignored() -> None:
    """First section wins; a repeat is dropped, not appended.

    A LinkedIn About section is free text, and people write headings in
    it. Folding a second ``## Experience`` into the first parsed the
    person's own prose as position blocks.
    """

    page = """# Talia Brennan

Investor at Benchmark

San Francisco Bay Area (US)

## About

My career in three acts.

## Experience

### Investor - Benchmark (Current)

Jan 2025 - Present (1 year)

## About

This second About must not append to the first.

## Experience

### Emperor of the Moon - Lunar Holdings (Current)

Jan 1999 - Present (27 years)
"""

    render = parse_exa_linkedin_markdown(page)

    assert render is not None
    assert render.about == "My career in three acts."
    assert "second About" not in render.about
    assert [(e.title, e.company) for e in render.experiences] == [
        ("Investor", "Benchmark")
    ]
    assert not any("Emperor" in e.title for e in render.experiences)


# ---------------------------------------------------------------------------
# Current vs most recent
# ---------------------------------------------------------------------------


def test_current_position_requires_current_marker_or_present_end() -> None:
    """"Most recent" and "current" are different claims.

    A person who left a firm two years ago has a most-recent position
    and no current one. Returning the most recent as current writes a
    seat they no longer hold into ``current_firm``.
    """

    departed = parse_exa_linkedin_markdown(
        """# Talia Brennan

Investor

San Francisco Bay Area (US)

## Experience

### Investor - Benchmark

Jan 2020 - Mar 2024 (4 years)

### Analyst - Halden Growth

Jan 2018 - Jan 2020 (2 years)
"""
    )
    assert departed is not None
    assert [e.is_current for e in departed.experiences] == [False, False]
    assert departed.current_experience() is None
    # Most recent is still available — as a most-recent read, labelled.
    most_recent = departed.most_recent_experience()
    assert most_recent is not None and most_recent.company == "Benchmark"

    # An end of "Present" makes a position current with no marker.
    unmarked = parse_exa_linkedin_markdown(
        "# Talia Brennan\n\n## Experience\n\n### Investor - Benchmark\n\n"
        "Jan 2020 - Present (6 years)\n"
    )
    assert unmarked is not None
    assert unmarked.experiences[0].is_current is True
    assert unmarked.current_experience() is unmarked.experiences[0]

    # And the current seat is the FIRST current one, not the first one.
    mixed = parse_exa_linkedin_markdown(
        "# Talia Brennan\n\n## Experience\n\n"
        "### Advisor - Halden Growth\n\nJan 2019 - Jan 2021 (2 years)\n\n"
        "### Investor - Benchmark (Current)\n\nJan 2022 - Present (4 years)\n"
    )
    assert mixed is not None
    current = mixed.current_experience()
    assert current is not None and current.company == "Benchmark"
    assert mixed.most_recent_experience().company == "Halden Growth"


def test_fixed_end_date_overrides_current_marker() -> None:
    """The dates the page STATES beat Exa's derived ``(Current)`` label."""

    render = parse_exa_linkedin_markdown(
        """# Talia Brennan

Investor

San Francisco Bay Area (US)

## Experience

### Investor - Benchmark (Current)

Jan 2020 - Mar 2024 (4 years and 2 months)
"""
    )

    assert render is not None
    stale = render.experiences[0]
    assert stale.end == "Mar 2024"
    assert stale.is_current is False
    assert render.current_experience() is None


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


def test_location_must_look_geographic() -> None:
    """A trailing ``(US)`` is a country code, not a geography test.

    A location is a filter input downstream, so a headline mistaken for
    one silently removes a real person from a search.
    """

    for good in (
        "San Francisco, California, United States",
        "San Francisco Bay Area (US)",
        "Greater Boston Area",
        "United States (US)",
        "California",
        "Brazil",
    ):
        assert looks_like_location(good), good

    for bad in (
        "Crypto Strategy at Visa (US)",
        "Partner at Greylock | Company Builder and Investor",
        "Investor & Advisor. Ex-Khosla Ventures, BlackRock",
        "Austin",  # a bare city with no region does not qualify
        "500 connections",
        "",
    ):
        assert not looks_like_location(bad), bad

    # And the header reader honours it: a non-geographic line ending in
    # a country code stays in the headline rather than becoming a place.
    render = parse_exa_linkedin_markdown(
        "# Talia Brennan\n\nCrypto Strategy at Visa (US)\n\n"
        "500 connections • 4,896 followers\n\n## Experience\n\n"
        "### Strategy - Visa (Current)\n\nJan 2024 - Present (2 years)\n"
    )
    assert render is not None
    assert render.location == ""
    assert render.headline == "Crypto Strategy at Visa (US)"


def test_location_helpers() -> None:
    assert strip_country_suffix("San Francisco Bay Area (US)") == (
        "San Francisco Bay Area"
    )
    assert strip_country_suffix("San Francisco, California, United States (US)") == (
        "San Francisco, California, United States"
    )
    assert looks_like_location("United States (US)")
    assert looks_like_location("San Francisco, California, United States")
    # Prose that merely mentions a place is not a location line.
    assert not looks_like_location(
        "Investor & Advisor. Ex-Khosla Ventures, Bain Capital Ventures, BlackRock"
    )
    assert not looks_like_location("")


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------


def test_parses_experiences_with_dates_and_current_flag() -> None:
    render = parse_exa_linkedin_markdown(INVESTOR_PAGE)
    assert render is not None

    assert [(e.title, e.company) for e in render.experiences] == [
        ("Investor", "Benchmark"),
        ("Research", "Helix Institute"),
        ("Member of Technical Staff", "Tensor Harbor"),
    ]

    current = render.experiences[0]
    assert current.is_current is True
    assert current.start == "Jan 2025"
    assert current.end == "Present"
    assert current.duration_text == "1 year and 7 months"
    assert current.location == "San Francisco, California, United States"
    assert current.department == "Consulting"
    assert current.level == ""
    assert current.company_url == "https://www.linkedin.com/company/benchmark-vc"

    past = render.experiences[1]
    assert past.is_current is False
    assert (past.start, past.end) == ("Jan 2025", "Jan 2025")
    # No duration parenthetical on this line; the location still parses.
    assert past.duration_text == ""
    assert past.location == "Stanford, California, United States"
    assert (past.department, past.level) == ("Research", "Specialist")

    # `current_experience` is the seat the record should carry.
    seat = render.current_experience()
    assert seat is not None and (seat.title, seat.company) == ("Investor", "Benchmark")


def test_parses_grouped_positions_under_one_company() -> None:
    """Two roles at one firm arrive as a company header plus ``####`` titles.

    Reading the group header as a position would mint a titleless job;
    reading the titles without it would mint firmless ones.
    """

    render = parse_exa_linkedin_markdown(OPERATOR_PAGE)
    assert render is not None

    assert [(e.title, e.company, e.is_current) for e in render.experiences] == [
        ("Principal", "Kiln Ventures", True),
        ("Advisor & Angel Investor", "Various Startups", True),
        ("Self-Serve Strategy", "Canvasly", False),
        ("Strategic Finance & Business Operations", "Canvasly", False),
    ]
    assert render.experiences[2].company_url == (
        "https://www.linkedin.com/company/canvasly"
    )
    assert render.experiences[3].start == "Jan 2018"
    assert render.experiences[3].end == "Dec 2020"
    # The company group header is not itself a position.
    assert all(e.title for e in render.experiences)


def test_ignores_firm_blurbs() -> None:
    """The paragraph under a position is about the FIRM, not the person.

    Rendering it as the person's evidence is how a record acquires
    accomplishments it never earned — Exa appends one to every block.
    """

    render = parse_exa_linkedin_markdown(INVESTOR_PAGE)
    assert render is not None

    flat = " ".join(
        " ".join(
            [e.title, e.company, e.location, e.department, e.level, e.duration_text]
        )
        for e in render.experiences
    )
    for blurb in ("40-50 employees", "annual revenue", "founded in 1995", "nonprofit"):
        assert blurb not in flat

    # A blurb also never leaks into the dates of the position below it.
    assert render.experiences[2].start == "Jan 2023"
    assert render.experiences[2].location == "New York, United States"

    # And a blurb between blocks does not become an extra position.
    assert len(render.experiences) == 3


# ---------------------------------------------------------------------------
# Position bullets
# ---------------------------------------------------------------------------


def test_keeps_position_bullets_between_date_and_meta_lines() -> None:
    """The person's own bullets are first-party evidence; the blurb is not.

    Before this the parser dropped every ``- `` line, so the judge read
    "no details" on a Team Lead seat whose page says the person led a pod
    and ran technical interviews.
    """

    render = parse_exa_linkedin_markdown(BULLETED_PAGE)
    assert render is not None
    by_title = {e.title: e for e in render.experiences}

    lead = by_title["Human Data Expert Team Lead"]
    assert lead.company == "SpaceXAI"
    assert lead.bullets == [
        "Wrote and conducted technical interviews to grow a team of Software Engineers.",
        "Led a pod (10-15 people) in solving technical issues related to improving LLM programming ability.",
        "Maintained internal tooling and coordinated contributions from direct reports.",
    ]
    # The meta line after the bullets still parses.
    assert (lead.department, lead.level) == ("Research", "Head")

    # Bullets on a single-firm block, whitespace collapsed and the marker dropped.
    assistant = by_title["Undergraduate Research Assistant"]
    assert assistant.bullets == [
        "Wrote Java and Python software to optimize workflows and allow for analysis of high-throughput imaging.",
        "Trained new hires on basic laboratory techniques.",
        "Department: Biology. Built the lab's imaging pipeline.",
    ]
    # A bullet that happens to read like metadata stays the person's sentence.
    assert (assistant.department, assistant.level) == ("Research", "Entry")

    # A position with no bullets has none, and never borrows a neighbour's.
    assert by_title["Data Operations"].bullets == []
    assert by_title["AI Coding Tutor"].bullets == []


def test_bullets_never_absorb_a_firm_blurb_or_a_line_after_it() -> None:
    render = parse_exa_linkedin_markdown(BULLETED_PAGE)
    assert render is not None

    every_bullet = " ".join(b for e in render.experiences for b in e.bullets)
    for blurb in ("employees", "Software Development company", "Technology, Information"):
        assert blurb not in every_bullet
    # The firm blurb closes the block: a dash line after it is nobody's bullet.
    assert "stray dash line" not in every_bullet


def test_keeps_position_prose_between_date_and_meta_lines() -> None:
    """A description written as prose is the person's words, not the firm's.

    Before this the parser read any non-bullet line as the firm blurb, so a
    Queue Manager seat whose page says the person leads contributor sessions
    and runs audits reached the judge as "no details".
    """

    render = parse_exa_linkedin_markdown(PROSE_PAGE)
    assert render is not None
    queue = render.experiences[0]
    assert (queue.title, queue.company) == ("Queue Manager (HireArt contractor)", "Scale AI")
    assert len(queue.bullets) == 1
    assert queue.bullets[0].startswith("As a Queue Manager, I ensure seamless project execution")
    assert "lead contributor sessions" in queue.bullets[0]
    assert "audits and feedback" in queue.bullets[0]
    # The meta line after the prose still parses.
    assert (queue.department, queue.level) == ("Operations", "Manager")


def test_prose_does_not_close_the_block() -> None:
    """Prose then bullets then the meta line: all three survive, in page order."""

    render = parse_exa_linkedin_markdown(PROSE_PAGE)
    assert render is not None
    lead = render.experiences[1]
    assert lead.title == "Delivery Lead"
    assert lead.bullets == [
        "Ran the coding-evaluation program for two frontier labs.",
        "Hired and managed 40 contract reviewers.",
        "Built the rubric audit loop.",
    ]
    assert (lead.department, lead.level) == ("Operations", "Lead")


def test_prose_never_absorbs_the_firm_blurb() -> None:
    """Text after the meta line is the blurb; with no meta line, the blurb shape ends it."""

    render = parse_exa_linkedin_markdown(PROSE_PAGE)
    assert render is not None
    every_line = " ".join(b for e in render.experiences for b in e.bullets)
    for blurb in ("employees", "Software Development company", "founded in", "delivers high quality"):
        assert blurb not in every_line, blurb

    # No Department/Level line: the prose is kept and the blurb after it is not.
    analyst = render.experiences[2]
    assert analyst.bullets == ["Wrote the weekly pipeline-quality report read by the COO."]
    # The blurb still closes the block.
    assert "stray dash line" not in every_line

    # No meta line and no blurb: the prose is the person's.
    assistant = render.experiences[3]
    assert assistant.bullets == ["Maintained the imaging pipeline and trained new hires."]


def test_prose_keeps_the_bullet_caps() -> None:
    long_line = "Led " + "the audit program " * 40
    lines = "\n\n".join(f"Paragraph {n}: shipped the rubric." for n in range(20))
    page = (
        "# Casey Lindqvist\n\n## Experience\n\n### Lead - Acme\n\n"
        "Jan 2024 - Present (2 years)\n\n"
        f"{long_line}\n\n{lines}\n\nDepartment: Operations • Level: Lead\n"
    )
    render = parse_exa_linkedin_markdown(page)
    assert render is not None
    seat = render.experiences[0]
    assert len(seat.bullets) == BULLETS_PER_POSITION_MAX
    assert len(seat.bullets[0]) == BULLET_MAX_CHARS
    assert seat.level == "Lead"


def test_markup_only_lines_are_not_kept_as_description() -> None:
    page = (
        "# Casey Lindqvist\n\n## Experience\n\n### Lead - Acme\n\n"
        "Jan 2024 - Present (2 years)\n\n####\n\n---\n\n> Ran the weekly audit.\n\n"
        "Department: Operations • Level: Lead\n"
    )
    render = parse_exa_linkedin_markdown(page)
    assert render is not None
    assert render.experiences[0].bullets == ["Ran the weekly audit."]


def test_a_sentence_that_mentions_level_is_prose_not_metadata() -> None:
    """Only a line that STARTS with Department:/Level: is the seat's meta line."""

    page = (
        "# Casey Lindqvist\n\n## Experience\n\n### Lead - Acme\n\n"
        "Jan 2024 - Present (2 years)\n\n"
        "Strategy and Direction: set the program's goals. Promoted to Level: Senior in 2025.\n\n"
        "Ran the weekly audit.\n\n"
        "Department: Operations • Level: Lead\n"
    )
    render = parse_exa_linkedin_markdown(page)
    assert render is not None
    seat = render.experiences[0]
    assert seat.bullets == [
        "Strategy and Direction: set the program's goals. Promoted to Level: Senior in 2025.",
        "Ran the weekly audit.",
    ]
    assert (seat.department, seat.level) == ("Operations", "Lead")


def test_bullets_round_trip_and_old_records_load_without_them() -> None:
    render = parse_exa_linkedin_markdown(BULLETED_PAGE)
    assert render is not None
    payload = render.to_dict()
    assert LinkedInRender.from_dict(payload) == render

    # A render stored before bullets existed loads with none.
    for experience in payload["experiences"]:
        experience.pop("bullets")
    reloaded = LinkedInRender.from_dict(payload)
    assert all(e.bullets == [] for e in reloaded.experiences)


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------


def test_parses_education() -> None:
    render = parse_exa_linkedin_markdown(OPERATOR_PAGE)
    assert render is not None

    assert render.education == [
        RenderedEducation(
            degree="M.B.A",
            school="Stanford Graduate School of Business",
            school_url="https://www.linkedin.com/school/stanford-gsb",
            years="1982 - 1984",
        ),
        RenderedEducation(
            degree="BA",
            school="Rice University",
            school_url="",
            years="1975 - 1979",
        ),
    ]

    # The section ends at the next ``##`` heading — an award is not a degree.
    assert not any("Wharton" in e.school for e in render.education)

    # A page with no education section says so, rather than guessing.
    investor = parse_exa_linkedin_markdown(INVESTOR_PAGE)
    assert investor is not None and investor.education == []


# ---------------------------------------------------------------------------
# Total experience
# ---------------------------------------------------------------------------


def test_total_experience_years() -> None:
    render = parse_exa_linkedin_markdown(INVESTOR_PAGE)
    assert render is not None
    assert render.total_experience_text == "4 years and 11 months"
    assert render.total_experience_years == 4.92

    operator = parse_exa_linkedin_markdown(OPERATOR_PAGE)
    assert operator is not None
    assert operator.total_experience_years == 14.67

    # No About section, so the page states no total. Absence is absence.
    bare = parse_exa_linkedin_markdown(HEADLINE_LESS_PAGE)
    assert bare is not None
    assert bare.total_experience_text == ""
    assert bare.total_experience_years is None

    assert duration_to_years("1 year") == 1.0
    assert duration_to_years("6 months") == 0.5
    assert duration_to_years("") is None
    assert duration_to_years("a while") is None


# ---------------------------------------------------------------------------
# Not a LinkedIn render
# ---------------------------------------------------------------------------


def test_non_linkedin_text_returns_none() -> None:
    """Neither a name header nor an Experience section: not this shape."""

    for text in (
        "",
        "   \n\n  ",
        "Cedarline Capital\nTeam\n\nTalia Brennan is Chief of Staff at Cedarline.",
        "## Portfolio\n\nWe back founders at the seed stage.",
    ):
        assert parse_exa_linkedin_markdown(text) is None

    # A name header alone is enough — a profile with no positions listed
    # still carries the person's name, headline and location.
    minimal = parse_exa_linkedin_markdown(
        "# Talia Brennan\n\nInvestor\n\nAustin, Texas, United States (US)\n"
    )
    assert minimal is not None
    assert minimal.name == "Talia Brennan"
    assert minimal.experiences == []


# ---------------------------------------------------------------------------
# Binding a render to a person
# ---------------------------------------------------------------------------


def test_render_names_match_accepts_variants_and_rejects_strangers() -> None:
    """Middle initials and diacritics are one person; a stranger is not."""

    for variant in (
        "Talia Brennan",
        "Talia S. Brennan",
        "Brennan, Talia",
        "Tália Brennan",
        "TALIA BRENNAN",
    ):
        assert render_names_match(variant, "Talia Brennan"), variant

    for stranger in ("Omar Vestergaard", "Talia Okonkwo", "Brennan Talbot", ""):
        assert not render_names_match(stranger, "Talia Brennan"), stranger

    # Empty on either side is never a match: "unknown" must not bind to
    # "unknown".
    assert not render_names_match("Talia Brennan", "")
    assert not render_names_match("", "")


def test_bind_render_drops_someone_elses_page() -> None:
    render = parse_exa_linkedin_markdown(INVESTOR_PAGE)
    assert render is not None and render.name == "Talia Brennan"

    assert bind_render(render, "Talia S. Brennan") is render
    assert bind_render(render, "Omar Vestergaard") is None
    assert bind_render(None, "Talia Brennan") is None


def test_seat_pair_moves_title_and_firm_together() -> None:
    """A title from one source paired with a firm from another is fiction."""

    render = parse_exa_linkedin_markdown(INVESTOR_PAGE)
    assert render is not None

    # A complete incoming pair stands untouched.
    assert seat_pair(render, "Chief of Staff", "Northwind") == (
        "Chief of Staff",
        "Northwind",
    )
    # Half a pair is replaced wholesale — never completed from the page.
    assert seat_pair(render, "Chief of Staff", "") == ("Investor", "Benchmark")
    assert seat_pair(render, "", "Northwind") == ("Investor", "Benchmark")
    assert seat_pair(render, "", "") == ("Investor", "Benchmark")
    # No render, or a render with no seat: whatever came in survives.
    assert seat_pair(None, "Chief of Staff", "") == ("Chief of Staff", "")
    empty = parse_exa_linkedin_markdown("# Talia Brennan\n\nInvestor\n")
    assert empty is not None and empty.experiences == []
    assert seat_pair(empty, "Chief of Staff", "") == ("Chief of Staff", "")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_render_round_trips_through_dict() -> None:
    """The render is stored on the candidate record, so it must survive JSON."""

    import json

    for page in (INVESTOR_PAGE, OPERATOR_PAGE, HEADLINE_LESS_PAGE):
        render = parse_exa_linkedin_markdown(page)
        assert render is not None

        payload = render.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert LinkedInRender.from_dict(payload) == render

        # A stored record written by an older build carries keys this
        # one does not know; loading must not explode on them.
        payload["retired_field"] = "ignored"
        payload["experiences"][0]["retired_field"] = "ignored"
        assert LinkedInRender.from_dict(payload) == render

    # An empty payload is a legitimate render, not a crash.
    assert LinkedInRender.from_dict({}) == LinkedInRender()
    assert RenderedExperience.from_dict({}) == RenderedExperience()
