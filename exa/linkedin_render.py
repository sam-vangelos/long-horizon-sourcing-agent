"""Parse Exa's structured LinkedIn Markdown into a profile.

Exa renders a name, optional headline and location, About, total experience,
experience, and education sections. This module parses those fields
deterministically; it does not infer absent data or call a model.

A missing section yields an empty value. Location remains empty unless the
render carries a valid location shape. Company-enrichment blurbs are excluded
from person evidence because they describe the firm rather than the person.
A position's own description is kept, whether written as ``- `` bullets or as
prose: it is the person's words about the seat, and it sits before the
``Department:``/``Level:`` line, after which comes the blurb.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from shared.identity_resolution import normalize_person_name


# A markdown inline link: ``[Benchmark](https://linkedin.com/company/x)``.
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")

# ``## Experience`` / ``## Education`` / ``## Patents`` — the section
# boundary. Everything between two of these belongs to the first.
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")

# A position or education header inside a section.
_H3_RE = re.compile(r"^###\s+(.+?)\s*$")
# A position nested under a company group header.
_H4_RE = re.compile(r"^####\s+(.+?)\s*$")

# The social-proof line under the location: "500 connections • 2,111
# followers", "1,637 followers". Never a headline, never a location.
_SOCIAL_RE = re.compile(
    r"^[\d,]+\+?\s+(?:connections?|followers)"
    r"(?:\s*[•·|]\s*[\d,]+\+?\s+(?:connections?|followers))*$",
    re.IGNORECASE,
)

# "Total Experience: 4 years and 11 months".
_TOTAL_EXPERIENCE_RE = re.compile(
    r"^Total\s+Experience:\s*(.+?)\s*$", re.IGNORECASE
)

# "4 years and 11 months", "1 year", "6 months", "2 years".
_YEARS_RE = re.compile(r"(\d+)\s*years?", re.IGNORECASE)
_MONTHS_RE = re.compile(r"(\d+)\s*months?", re.IGNORECASE)

# A position's date line, the line that immediately follows its header:
#   "Jan 2025 - Present (1 year and 7 months) in San Francisco, ..."
#   "Jan 2025 - Jan 2025 in Stanford, California, United States"
#   "Feb 2022 - Present (4 years and 6 months)"
_DATE_TOKEN = r"(?:[A-Z][a-z]{2,8}\.?\s+\d{4}|\d{4}|Present|present)"
_DATE_LINE_RE = re.compile(
    rf"^(?P<start>{_DATE_TOKEN})\s*[-–—]\s*(?P<end>{_DATE_TOKEN})"
    r"(?:\s*\((?P<duration>[^)]*)\))?"
    r"(?:\s+in\s+(?P<location>.+?))?\s*$"
)

# An education entry's year line: "1997 - 1999 (2 years)", "2020 - 2024".
_EDU_YEARS_RE = re.compile(
    r"^(?P<range>\d{4}\s*[-–—]\s*(?:\d{4}|Present|present))"
    r"(?:\s*\([^)]*\))?\s*$"
)

# A position bullet the person wrote: "- Led a pod (10-15 people) ...". Exa
# writes them with a hyphen; the other list markers are accepted because
# LinkedIn's own text sometimes carries them through.
_BULLET_RE = re.compile(r"^[-*•·]\s+(?P<text>.*)$")

# Leading markup left inside a bullet's text after its marker ("- ## x",
# "- > x"), stripped so a page cannot dress a bullet up as structure.
_BULLET_LEADING_MARKUP_RE = re.compile(r"^(?:[#*>•·]+\s*|[-–—]\s+)+")

# Bounds on one position's description lines, bullets and prose together.
# The page excerpt is already capped at ingestion; these keep one verbose
# position from crowding out the rest of the career in the judge's prompt.
BULLET_MAX_CHARS = 400
BULLETS_PER_POSITION_MAX = 12

# "Department: Consulting • Level: Specialist" — either half may be absent.
_DEPARTMENT_RE = re.compile(r"Department:\s*([^•·|]+)")
_LEVEL_RE = re.compile(r"Level:\s*([^•·|]+)")
# The meta line STARTS with one of its labels. In the stored Exa probe corpus
# 5,715 of 5,716 lines carrying either label start with it; the other is a
# sentence of the person's own description that mentions one mid-line.
_META_LINE_RE = re.compile(r"^(?:Department|Level):")

# The shape of Exa's firm blurb: "<Firm> is a <Industry> company.", "<Firm>
# has 200-300 employees", "Headquartered in ...", "founded in 2016". Read only
# for a position with NO meta line, where nothing else marks where the
# person's description ends and the firm's begins. 3,528 of the 3,567 blurbs
# that follow a meta line in the stored Exa probe corpus match it.
_BLURB_SHAPE_RE = re.compile(
    r"^.{1,120}?\bis (?:a|an|the)\b[^.]{0,120}\b(?:company|organi[sz]ation|firm|"
    r"agency|institution|university|college|school|startup|non-?profit|business|"
    r"practice|partnership|government)\b"
    r"|\bhas [\d,+]+(?:-[\d,+]+)? employees\b"
    r"|\bHeadquartered in\b"
    r"|\bfounded in \d{4}\b",
    re.IGNORECASE,
)

# A line with no letter or digit carries nothing to keep ("####", "---").
_HAS_WORD_RE = re.compile(r"\w")

# A trailing ISO-ish country code on the location line: "... (US)".
_COUNTRY_SUFFIX_RE = re.compile(r"\s*\(([A-Z]{2,3})\)\s*$")

# A proper-noun run for the "City, Region[, Country]" location shape. Keep this
# aligned with ``exa/extraction.py:_LOCATION_LINE_RE``.
_PROPER_RUN = r"[A-Z][A-Za-z.'’-]*(?:\s[A-Za-z.'’-]+)*"
_LOCATION_SHAPE_RE = re.compile(
    rf"^{_PROPER_RUN},\s{_PROPER_RUN}(?:,\s{_PROPER_RUN})?$"
)

# The marker Exa puts on a position the person still holds.
_CURRENT_MARKER_RE = re.compile(r"\s*\(Current\)\s*$", re.IGNORECASE)

# Split "<Title> - <Firm>" headings like ``exa/extraction.py:_SEGMENT_SPLIT_RE``;
# surrounding space is load-bearing so bare hyphens stay inside names.
_TITLE_FIRM_SPLIT_RE = re.compile(r"\s+[-–—]\s+")

# Longest string that can plausibly be a location label. Same value and
# same reason as ``exa.extraction.LOCATION_MAX_CHARS``.
LOCATION_MAX_CHARS = 60

MONTHS_PER_YEAR = 12

# Whole-label places: a location line with no comma and no trailing
# "Area" qualifies only by being one of these outright. US states
# because LinkedIn writes them bare, and the countries a cross-border
# sourcing campaign actually meets. Lowercased for comparison.
_US_STATES = (
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
    "delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|"
    "kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
    "mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
    "new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|"
    "pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|"
    "utah|vermont|virginia|washington|west virginia|wisconsin|wyoming|"
    "district of columbia|washington dc|puerto rico"
).split("|")

_COUNTRIES = (
    "united states|united kingdom|canada|mexico|brazil|argentina|chile|"
    "colombia|peru|ireland|france|germany|spain|portugal|italy|netherlands|"
    "belgium|switzerland|austria|sweden|norway|denmark|finland|poland|"
    "czechia|czech republic|romania|ukraine|greece|turkey|israel|"
    "united arab emirates|india|pakistan|bangladesh|china|hong kong|taiwan|"
    "japan|south korea|singapore|malaysia|indonesia|thailand|vietnam|"
    "philippines|australia|new zealand|south africa|nigeria|kenya|egypt|"
    "estonia|latvia|lithuania|hungary|bulgaria|serbia|croatia|slovakia|"
    "slovenia|iceland|luxembourg|russia|saudi arabia|qatar|morocco"
).split("|")

_PLACE_NAMES = frozenset(
    place.strip() for place in (*_US_STATES, *_COUNTRIES) if place.strip()
)


# ---------------------------------------------------------------------------
# Rendered pieces
# ---------------------------------------------------------------------------


@dataclass
class RenderedExperience:
    """One position block, exactly as the page wrote it."""

    title: str = ""
    company: str = ""
    company_url: str = ""
    start: str = ""
    end: str = ""
    is_current: bool = False
    location: str = ""
    department: str = ""
    level: str = ""
    duration_text: str = ""
    # The person's own description of this position, one entry per line in
    # page order: ``- `` bullets with the marker dropped, and prose lines,
    # whitespace collapsed. Never the firm blurb. Additive with a default, so
    # a render stored before descriptions were parsed loads with none.
    bullets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RenderedExperience:
        payload = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        bullets = payload.get("bullets")
        payload["bullets"] = (
            [str(b) for b in bullets if isinstance(b, str) and b.strip()]
            if isinstance(bullets, list)
            else []
        )
        return cls(**payload)


@dataclass
class RenderedEducation:
    """One education block. ``years`` is the raw range, "1997 - 1999"."""

    degree: str = ""
    school: str = ""
    school_url: str = ""
    years: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RenderedEducation:
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class LinkedInRender:
    """A parsed Exa LinkedIn profile page.

    Every field is what the page said or empty. ``experiences`` and
    ``education`` are in PAGE ORDER, which on LinkedIn is
    reverse-chronological — the parser does not re-sort, because a
    reordering is a claim about dates the page may not carry.
    """

    name: str = ""
    headline: str = ""
    location: str = ""
    about: str = ""
    total_experience_text: str = ""
    total_experience_years: float | None = None
    experiences: list[RenderedExperience] = field(default_factory=list)
    education: list[RenderedEducation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LinkedInRender:
        payload = {
            k: v
            for k, v in (d or {}).items()
            if k in cls.__dataclass_fields__ and k not in ("experiences", "education")
        }
        payload["experiences"] = [
            RenderedExperience.from_dict(e) for e in (d or {}).get("experiences") or []
        ]
        payload["education"] = [
            RenderedEducation.from_dict(e) for e in (d or {}).get("education") or []
        ]
        return cls(**payload)

    def current_experience(self) -> RenderedExperience | None:
        """The first position the page says the person STILL HOLDS, or None.

        No fallback. "Most recent" and "current" are different claims,
        and a person who left a firm two years ago has a most-recent
        position and no current one — returning it here would write a
        seat they no longer hold into ``current_firm``, which is the
        stale-title failure this source already has to caveat.

        Use :meth:`most_recent_experience` when a most-recent read is
        what is wanted, and label it as such.
        """

        for experience in self.experiences:
            if experience.is_current:
                return experience
        return None

    def most_recent_experience(self) -> RenderedExperience | None:
        """The first position in page order — LinkedIn's own newest-first.

        A most-recent read, NOT a current-seat claim. Callers that fill
        a blank title and firm from it are recording where the page last
        placed this person, not asserting they are there now.
        """

        return self.experiences[0] if self.experiences else None


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


def _split_link(text: str) -> tuple[str, str]:
    """``"[Benchmark](url)"`` -> ``("Benchmark", "url")``; plain text passes through."""

    match = _LINK_RE.search(text or "")
    if not match:
        return _clean(text), ""
    label = _clean(match.group(1))
    url = _clean(match.group(2))
    # A link inside a longer string: keep the surrounding words, drop the
    # markup, so "Advisor at [Canvasly](url) (part-time)" stays readable.
    remainder = _LINK_RE.sub(label, text)
    return _clean(remainder), url


def duration_to_years(text: str) -> float | None:
    """"4 years and 11 months" -> 4.92; None when nothing parses.

    Rounded to two places because the page's own precision is months.
    """

    raw = str(text or "")
    years_match = _YEARS_RE.search(raw)
    months_match = _MONTHS_RE.search(raw)
    if not years_match and not months_match:
        return None
    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    return round(years + months / MONTHS_PER_YEAR, 2)


def strip_country_suffix(location: str) -> str:
    """"San Francisco Bay Area (US)" -> "San Francisco Bay Area"."""

    return _clean(_COUNTRY_SUFFIX_RE.sub("", str(location or "")))


def looks_like_location(line: str) -> bool:
    """True only for a line that reads as a PLACE.

    A trailing ``(US)`` was accepting anything that carried one, which
    is a country-code test rather than a geography test. A location is a
    filter input downstream, so a headline mistaken for one silently
    removes a real person from a search.

    Three ways to qualify, all structural:

    * a comma — "San Francisco, California, United States";
    * a trailing "Area" — "San Francisco Bay Area", "Greater Boston Area";
    * the whole label is a US state or a country — "United States",
      "California", "Brazil".

    A bare city with no region ("Austin") does NOT qualify. That is
    deliberate: absence costs breadth, a wrong location costs the
    candidate.
    """

    text = strip_country_suffix(_clean(line))
    if not text or len(text) > LOCATION_MAX_CHARS:
        return False
    if "," in text:
        # Still has to read as proper nouns, not as prose with a comma.
        return bool(_LOCATION_SHAPE_RE.match(text))
    if text.lower().endswith(" area"):
        return True
    return text.lower() in _PLACE_NAMES


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------


def _sections(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Split the page into the header block and its FIRST ``##`` sections.

    A heading that repeats is DISCARDED, not appended to the first
    block. A person's About text can itself contain a line reading
    ``## Experience`` — LinkedIn About sections are free text and people
    write headings in them — and folding that into the real Experience
    section would parse the person's own prose as position blocks.
    First occurrence wins; everything under a repeat goes to a sink.

    Section names fold to lowercase, so ``## about`` and ``## About``
    are one section rather than two.
    """

    header: list[str] = []
    sections: dict[str, list[str]] = {}
    sink: list[str] = []
    current: list[str] | None = None

    for line in lines:
        match = _SECTION_RE.match(line)
        if match:
            key = _clean(match.group(1)).lower()
            if key in sections:
                # A repeat. Everything under it is dropped on the floor.
                sink = []
                current = sink
            else:
                current = sections.setdefault(key, [])
            continue
        (header if current is None else current).append(line)

    return header, sections


def _parse_header(header_lines: list[str]) -> tuple[str, str, str]:
    """Name, headline, and location off the top of the page.

    The headline is OPTIONAL — 8 of the 40 profiles in the live run had
    none, and the location line sat directly under the name. So the
    location is found by SHAPE (the last shaped line above the
    connections count) rather than by position, and the headline is
    whatever non-social line is left. Reading position alone is what
    would put "San Francisco Bay Area (US)" into a person's headline.
    """

    name = ""
    body: list[str] = []
    for raw in header_lines:
        line = _clean(raw)
        if not line:
            continue
        if not name and line.startswith("# "):
            name = _clean(line[2:])
            continue
        if _SOCIAL_RE.match(line):
            # Everything below the social-proof line is page chrome.
            break
        body.append(line)

    location = ""
    for line in reversed(body):
        if looks_like_location(line):
            location = line
            break

    headline = next((line for line in body if line != location), "")
    return name, headline, location


def _parse_about(about_lines: list[str], whole_text: str) -> tuple[str, str]:
    """The About prose, and the ``Total Experience`` line's value.

    ``Total Experience`` is rendered INSIDE the About block, but it is a
    computed field rather than the person's own words, so it is lifted
    out of the prose and returned separately. It is also searched for
    across the whole page, because a profile with no About section still
    carries it.
    """

    prose: list[str] = []
    total = ""
    for raw in about_lines:
        line = _clean(raw)
        if not line:
            if prose and prose[-1] != "":
                prose.append("")
            continue
        match = _TOTAL_EXPERIENCE_RE.match(line)
        if match:
            total = _clean(match.group(1))
            continue
        prose.append(line)

    if not total:
        for raw in whole_text.splitlines():
            match = _TOTAL_EXPERIENCE_RE.match(_clean(raw))
            if match:
                total = _clean(match.group(1))
                break

    return "\n".join(prose).strip(), total


def _apply_date_line(experience: RenderedExperience, line: str) -> bool:
    match = _DATE_LINE_RE.match(line)
    if not match:
        return False
    experience.start = _clean(match.group("start"))
    experience.end = _clean(match.group("end"))
    experience.duration_text = _clean(match.group("duration") or "")
    experience.location = _clean(match.group("location") or "")
    # The DATES win over the marker, in both directions. An end of
    # "Present" makes a position current even with no ``(Current)``
    # label; a fixed end date makes it past even WITH one — Exa's marker
    # is derived, the date range is what the page states, and a stale
    # marker on a role that ended in 2024 would otherwise be written
    # into someone's current seat.
    experience.is_current = experience.end.lower() == "present"
    return True


def _bullet_text(line: str) -> str | None:
    """The text of a ``- `` bullet line, or None when the line is not one.

    An empty bullet ("- ") is a bullet with no text: "" rather than None,
    so the caller consumes the line without recording anything.
    """

    match = _BULLET_RE.match(line)
    if not match:
        return None
    text = _BULLET_LEADING_MARKUP_RE.sub("", _clean(match.group("text")))
    return text[:BULLET_MAX_CHARS]


def _description_text(line: str) -> str:
    """One description line as the judge reads it: a bullet or a prose line.

    Leading markup is stripped from prose the same way it is from a bullet,
    and a line left with no word in it ("####") is "" — consumed, not kept.
    """

    bullet = _bullet_text(line)
    if bullet is not None:
        text = bullet
    else:
        text = _BULLET_LEADING_MARKUP_RE.sub("", _clean(line))[:BULLET_MAX_CHARS]
    return text if _HAS_WORD_RE.search(text) else ""


def _keep_description(experience: RenderedExperience, line: str) -> None:
    text = _description_text(line)
    if text and len(experience.bullets) < BULLETS_PER_POSITION_MAX:
        experience.bullets.append(text)


def _is_meta_line(line: str) -> bool:
    return bool(_META_LINE_RE.match(line))


def _apply_meta_line(experience: RenderedExperience, line: str) -> bool:
    """``Department: Consulting • Level: Specialist`` — either half optional."""

    if not _is_meta_line(line):
        return False
    department = _DEPARTMENT_RE.search(line)
    level = _LEVEL_RE.search(line)
    if not department and not level:
        return False
    if department:
        experience.department = _clean(department.group(1))
    if level:
        experience.level = _clean(level.group(1))
    return True


def _parse_experience(section_lines: list[str]) -> list[RenderedExperience]:
    """Position blocks, in page order.

    Two block shapes, both live:

    * ``### <Title> - [<Firm>](url) (Current)`` — one position, one firm.
    * ``### [<Firm>](url)`` followed by ``#### <Title> (Current)`` — two
      or more positions at ONE firm. The group header carries the firm
      and every ``####`` under it inherits it. Reading the group header
      as a position would mint a titleless job; reading the ``####``
      titles without it would mint firmless ones.

    Each position's lines are gathered first and read as a block (see
    :func:`_fill_position`), because where the person's description ends
    depends on whether a ``Department:``/``Level:`` line follows it.
    """

    # Every position with its own lines, in page order. A company group
    # header owns no lines: anything between it and its first ``####`` is
    # dropped, as it always was.
    blocks: list[tuple[RenderedExperience, list[str]]] = []
    body_lines: list[str] | None = None
    group_company = ""
    group_company_url = ""

    for raw in section_lines:
        line = _clean(raw)
        if not line:
            continue

        h3 = _H3_RE.match(line)
        h4 = _H4_RE.match(line) if not h3 else None

        if h3 or h4:
            pending: RenderedExperience | None = None
            body = (h3 or h4).group(1)
            is_current = bool(_CURRENT_MARKER_RE.search(body))
            body = _CURRENT_MARKER_RE.sub("", body).strip()
            label, url = _split_link(body)

            if h4:
                # Nested under a company group: the heading is the title.
                pending = RenderedExperience(
                    title=label,
                    company=group_company,
                    company_url=group_company_url,
                    is_current=is_current,
                )
            elif _TITLE_FIRM_SPLIT_RE.search(label):
                # "Investor - Benchmark". The separator needs surrounding
                # space: a bare hyphen splits "Co-founder" and "Ex-Google".
                title, company = _TITLE_FIRM_SPLIT_RE.split(label, maxsplit=1)
                title, company = _clean(title), _clean(company)
                group_company, group_company_url = "", ""
                pending = RenderedExperience(
                    title=title,
                    company=company,
                    company_url=url,
                    is_current=is_current,
                )
            else:
                # A bare ``### <Firm>`` opens a company group. The firm
                # is recorded; the positions arrive as ``####`` lines.
                group_company, group_company_url = label, url
            if pending is None:
                body_lines = None
            else:
                body_lines = []
                blocks.append((pending, body_lines))
            continue

        if body_lines is not None:
            body_lines.append(line)

    for experience, lines in blocks:
        _fill_position(experience, lines)
    return [experience for experience, _lines in blocks]


def _fill_position(experience: RenderedExperience, lines: list[str]) -> None:
    """Dates, the person's description, and the meta line, from one block.

    Exa writes a position as: the date line, the person's own description
    (``- `` bullets, prose, or both), the ``Department:``/``Level:`` line,
    then the firm blurb. So:

    * Every line before the meta line is the person's, prose and bullets
      alike, kept on ``bullets`` in page order. Prose does not close the
      block: bullets and the meta line after it still parse.
    * The meta line is applied. Bullets directly after it are still kept;
      the first other line is the firm blurb, which closes the block, so a
      dash line after a blurb belongs to no position.
    * With NO meta line, nothing marks the end of the description, so it
      runs until the first line shaped like a firm blurb
      (``_BLURB_SHAPE_RE``), which closes the block.

    A bullet is tested before the meta line: a bullet that mentions
    "Level:" is still the person's sentence, not the seat's metadata.
    """

    meta_at = next(
        (
            index
            for index, line in enumerate(lines)
            if _bullet_text(line) is None and _is_meta_line(line)
        ),
        None,
    )
    described = lines if meta_at is None else lines[:meta_at]
    for line in described:
        if _apply_date_line(experience, line):
            continue
        if (
            meta_at is None
            and _bullet_text(line) is None
            and _BLURB_SHAPE_RE.search(line)
        ):
            return
        _keep_description(experience, line)
    if meta_at is None:
        return

    for line in lines[meta_at:]:
        if _apply_date_line(experience, line):
            continue
        if _bullet_text(line) is not None:
            _keep_description(experience, line)
            continue
        if _apply_meta_line(experience, line):
            continue
        # The firm blurb. The block is over.
        return


def _parse_education(section_lines: list[str]) -> list[RenderedEducation]:
    """Education blocks: ``### <Degree> at <School>`` plus a year range.

    A heading with no ``" at "`` is read as the SCHOOL, not the degree:
    under an Education heading the institution is the fact that is
    always present, and a degree is the optional detail.
    """

    education: list[RenderedEducation] = []
    pending: RenderedEducation | None = None
    open_for_detail = False

    def close() -> None:
        nonlocal pending
        if pending is not None:
            education.append(pending)
            pending = None

    for raw in section_lines:
        line = _clean(raw)
        if not line:
            continue

        heading = _H3_RE.match(line) or _H4_RE.match(line)
        if heading:
            close()
            degree_part, separator, school_part = heading.group(1).partition(" at ")
            if separator:
                degree = _clean(degree_part)
                school, url = _split_link(school_part)
            else:
                degree = ""
                school, url = _split_link(degree_part)
            pending = RenderedEducation(degree=degree, school=school, school_url=url)
            open_for_detail = True
            continue

        if pending is not None and open_for_detail:
            years = _EDU_YEARS_RE.match(line)
            if years:
                pending.years = _clean(years.group("range"))
                continue
            open_for_detail = False

    close()
    return education


# ---------------------------------------------------------------------------
# Binding a render to a person
# ---------------------------------------------------------------------------

_NAME_PUNCT = re.compile(r"[^a-z0-9\s]+")

# Why a page that IS a LinkedIn render was not attached to a record.
# Structural, like ``verify_before_outreach``: a downstream surface
# reads the field, it does not grep prose for the word "mismatch".
RENDER_NAME_MISMATCH_NOTE = "render_name_mismatch"


def _name_keys(name: str) -> set[str]:
    """Return normalized name keys used for render binding.

    Keep this local to avoid importing ``exa.identity`` through its schema
    dependency. Full, first/last, and comma-reordered keys are accepted; names
    are diacritic-folded and passed through
    ``shared.identity_resolution.normalize_person_name``.
    """

    decomposed = unicodedata.normalize("NFKD", str(name or ""))
    raw = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    tokens = _NAME_PUNCT.sub(" ", normalize_person_name(raw)).split()
    if not tokens:
        return set()

    keys = {" ".join(tokens)}
    if len(tokens) >= 2:
        keys.add(f"{tokens[0]} {tokens[-1]}")
        if "," in raw:
            reordered = tokens[1:] + tokens[:1]
            keys.add(" ".join(reordered))
            keys.add(f"{reordered[0]} {reordered[-1]}")
    return keys


def render_names_match(page_name: str, person_name: str) -> bool:
    """True when a render's ``# Name`` header IS this candidate.

    The binding S7 shipped without. A record's evidence list is not
    guaranteed to hold that person's own profile — a merge, a
    mis-attributed URL, or a page that redirected puts somebody else's
    LinkedIn page on the record, and parsing it would graft a stranger's
    entire career onto this candidate. That is worse than the thin
    profile S7 set out to fix: a thin record understates, a bound-wrong
    record fabricates.

    An empty name on either side is NOT a match. "Unknown" must never
    bind to "unknown".
    """

    page_keys = _name_keys(page_name)
    person_keys = _name_keys(person_name)
    if not page_keys or not person_keys:
        return False
    return bool(page_keys & person_keys)


def bind_render(render: LinkedInRender | None, person_name: str) -> LinkedInRender | None:
    """The render if it is this person's page, else None."""

    if render is None:
        return None
    return render if render_names_match(render.name, person_name) else None


def seat_pair(
    render: LinkedInRender | None, title: str, firm: str
) -> tuple[str, str]:
    """Title and firm, ATOMICALLY, from ONE source.

    A title read off the page title paired with a firm read off the
    render is a seat that appears on neither — "Chief of Staff at
    Benchmark", where the page calls the person an Investor at Benchmark
    and the title calls them a Chief of Staff somewhere else. It reads
    as a fact and is a composition, which is the fabrication class this
    module exists to prevent.

    So the pair moves together. A COMPLETE pair stands. Anything less —
    one half, or neither — is replaced wholesale by the render's seat
    when the render has one, and otherwise left exactly as it was.

    The render's seat is its CURRENT position when it has one, else its
    most recent. A most-recent read is the weaker claim, and the
    staleness caveat every Exa record carries is what states that; what
    matters here is that both halves come from ONE position block.
    """

    if title and firm:
        return title, firm
    if render is None:
        return title, firm
    seat = render.current_experience() or render.most_recent_experience()
    if seat is None or not (seat.title or seat.company):
        return title, firm
    return seat.title, seat.company


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_exa_linkedin_markdown(text: str) -> LinkedInRender | None:
    """Parse Exa's LinkedIn profile rendering, or None if it isn't one.

    None means "this text is not a LinkedIn profile render" — no ``#``
    name header AND no ``## Experience`` section. That is an ordinary
    outcome (a company page, a Pulse article, a plain-text scrape), not
    an error, and the caller keeps whatever it already had.
    """

    raw = str(text or "")
    if not raw.strip():
        return None

    lines = raw.splitlines()
    header_lines, sections = _sections(lines)
    has_experience = "experience" in sections
    has_name_header = any(_clean(line).startswith("# ") for line in header_lines)
    if not has_experience and not has_name_header:
        return None

    name, headline, location = _parse_header(header_lines)
    about, total_experience_text = _parse_about(sections.get("about", []), raw)

    return LinkedInRender(
        name=name,
        headline=headline,
        location=location,
        about=about,
        total_experience_text=total_experience_text,
        total_experience_years=duration_to_years(total_experience_text),
        experiences=_parse_experience(sections.get("experience", [])),
        education=_parse_education(sections.get("education", [])),
    )


__all__ = [
    "BULLETS_PER_POSITION_MAX",
    "BULLET_MAX_CHARS",
    "LOCATION_MAX_CHARS",
    "RENDER_NAME_MISMATCH_NOTE",
    "LinkedInRender",
    "RenderedEducation",
    "RenderedExperience",
    "bind_render",
    "duration_to_years",
    "looks_like_location",
    "parse_exa_linkedin_markdown",
    "render_names_match",
    "seat_pair",
    "strip_country_suffix",
]
