"""Match a candidate's employers against a brief's employer and alumni blacklists.

Pure text matching with no model call, shared by the LinkedIn facial gate, the
LinkedIn post-profile gate, and the Exa pre-judgment gate so every source
excludes the same people.

Two rules live here.

The EMPLOYER blacklist (:func:`match_current_employer`) keys on the CURRENT
employer. A current-company field matches on a word boundary. A headline
matches only where it names the employer ("@ X", "at X"): a bare mention is
usually a tool, certification, or skill. A past employer stays in the pool:
"ex-X", "Former engineer at X", "Ex-SWE @ X", "Prev. Data Lead at X". A past
marker covers the "at X" that follows it in the same clause; a clause ends at
"|", "·", ";", ",", or " - ".

The ALUMNI blacklist (:func:`match_any_employer`) keys on ANY employer, current
or past: anyone who has ever worked at a listed employer is never a lead. Company fields from the work history match on a word
boundary; a snippet's "Title at Company (dates)" entries, a headline, and an
About section match where they name the employer ("at X", "@ X") or mark it
past ("ex-X", "Former X", "X alum"), whatever the tense.

Both rules skip a name that is part of an institution or a concept rather than
an employer: "Babbage School", "Babbage Award", "Babbage engine", "Babbage Medal",
"Charles Babbage".
"""

from __future__ import annotations

import re
from typing import Iterable

# The words after a matched name that make it an institution or a concept
# rather than an employer: "Babbage School", "Babbage College", "Babbage Award",
# "Babbage engine", "Babbage-engine", "The Charles Babbage Institute".
_NON_EMPLOYER_NEXT_WORD_MARKERS = frozenset(
    {
        "school",
        "college",
        "university",
        "award",
        "awards",
        "prize",
        "medal",
        "engine",
        "engines",
        "machine",
        "machines",
        "institute",
        "fellowship",
        "lecture",
        "society",
        "archive",
    }
)

# The words before a matched name that make it someone else's: "Charles Babbage"
# (the person, and the institute and archive named after him).
_NON_EMPLOYER_PREVIOUS_WORD_MARKERS = frozenset({"charles"})

# Words that make the employer after them a past one: "ex", "former",
# "prev.", "past", "late of", "alum", and their longer forms.
_PAST_EMPLOYER_MARKER = re.compile(
    r"(?i)\b(?:ex|former|formerly|previously|prev|past|late\s+of"
    r"|alumni|alumnus|alum)\b"
)

# A past marker directly in front of the name, joined by a hyphen, a dot, a
# colon, or whitespace: "ex-Babbage", "Ex Babbage", "Former Babbage", "Prev. Babbage".
_PAST_MARKER_BEFORE_NAME = re.compile(
    r"(?i)\b(?:ex|former|formerly|previously|prev|past)\s*[-.:]?\s*$"
)

# An alumni word directly after the name: "Babbage alum", "Babbage alumni".
_ALUMNI_WORD_AFTER_NAME = re.compile(r"(?i)^\s*(?:alum|alumni|alumnus|alumna)\b")

# Separators between clauses: "|", "·", ";", ",", " - " and its dashes. A marker
# in an earlier clause ("ex-Google | Head of Data at X") says nothing about X.
_CLAUSE_SEPARATOR = re.compile(r"[|·;,]|\s[-–—]\s")

# What ends a past stretch inside a clause: "now" / "currently" ("Ex-Google
# now Head of Data at X"), or an earlier employer anchor, which the marker
# before it already described ("Former engineer at Google and Lead at X").
_PAST_STRETCH_END = re.compile(r"(?i)\b(?:now|currently)\b|\bat\b|@")

# How far back from the "at" / "@" anchor a marker may sit. A marker further
# back still counts when it opens the clause ("Former Senior Staff ...
# Engineer at X").
_LOOKBACK_CHARS = 32


def _names(blacklist: Iterable[str] | None) -> list[str]:
    if not blacklist or isinstance(blacklist, (str, bytes)):
        return []
    try:
        entries = list(blacklist)
    except TypeError:
        return []
    return [str(entry).strip() for entry in entries if str(entry or "").strip()]


def _matches(blocked: str, field_value: str) -> Iterable[re.Match[str]]:
    pattern = r"\b" + re.escape(blocked) + r"\b"
    return re.finditer(pattern, field_value, re.IGNORECASE)


def match_current_employer(
    current_company: str | None,
    headline: str | None,
    employer_blacklist: Iterable[str] | None,
) -> tuple[str, str, str] | None:
    """Return ``(blocked_name, field_name, field_value)`` or None.

    ``field_name`` is ``"current_company"`` or ``"headline"``. Blacklist order
    wins over field order: the first blacklisted name that matches either field
    is the one reported.
    """

    if not employer_blacklist:
        return None

    fields = (
        ("current_company", current_company or ""),
        ("headline", headline or ""),
    )
    for blocked_text in _names(employer_blacklist):
        for field_name, field_value in fields:
            if not field_value:
                continue
            for match in _matches(blocked_text, field_value):
                if _is_non_employer_context(field_value, match):
                    continue
                if field_name == "headline" and not _headline_names_employer(
                    field_value, match
                ):
                    continue
                return blocked_text, field_name, field_value
    return None


def match_any_employer(
    alumni_blacklist: Iterable[str] | None,
    *,
    current_company: str | None = None,
    headline: str | None = None,
    experience_companies: Iterable[str] | None = None,
    experience_entries: Iterable[str] | None = None,
    about: str | None = None,
) -> tuple[str, str, str] | None:
    """Return ``(blocked_name, field_name, field_value)`` or None.

    Matches a blacklisted name as ANY employer, current or past.
    ``field_name`` is one of ``"current_company"``, ``"experience"`` (a
    company field from the work history: a full profile's positions, an Exa
    render's positions), ``"experience_entry"`` (a search card's "Title at
    Company (dates)" line), ``"headline"``, or ``"about"``.

    Company fields match on a word boundary. The free-text fields match only
    where they name the name as an employer ("at X", "@ X") or as a past one
    ("ex-X", "Former X", "X alum"); a bare mention is a topic, not a job.
    Blacklist order wins over field order.
    """

    names = _names(alumni_blacklist)
    if not names:
        return None

    company_fields: list[tuple[str, str]] = []
    text_fields: list[tuple[str, str]] = []
    if current_company:
        company_fields.append(("current_company", str(current_company)))
    for company in experience_companies or ():
        if company:
            company_fields.append(("experience", str(company)))
    for entry in experience_entries or ():
        if entry:
            text_fields.append(("experience_entry", str(entry)))
    if headline:
        text_fields.append(("headline", str(headline)))
    if about:
        text_fields.append(("about", str(about)))

    for blocked_text in names:
        for field_name, field_value in company_fields:
            for match in _matches(blocked_text, field_value):
                if not _is_non_employer_context(field_value, match):
                    return blocked_text, field_name, field_value
        for field_name, field_value in text_fields:
            for match in _matches(blocked_text, field_value):
                if _is_non_employer_context(field_value, match):
                    continue
                if _text_names_employer_any_tense(field_value, match):
                    return blocked_text, field_name, field_value
    return None


def _is_non_employer_context(field_value: str, match: re.Match[str]) -> bool:
    remainder = field_value[match.end():]
    next_word = re.match(r"\W+([A-Za-z]+)\b", remainder)
    if next_word and next_word.group(1).lower() in _NON_EMPLOYER_NEXT_WORD_MARKERS:
        return True
    previous_word = re.search(r"\b([A-Za-z]+)\W+$", field_value[: match.start()])
    return bool(
        previous_word
        and previous_word.group(1).lower() in _NON_EMPLOYER_PREVIOUS_WORD_MARKERS
    )


def _anchor_length(prefix: str) -> int:
    """Length of the "at" / "@" anchor that ends ``prefix``, or 0 for none."""

    if prefix.endswith("@"):
        return 1
    if re.search(r"(?i)\bat$", prefix):
        return 2
    return 0


def _headline_names_employer(field_value: str, match: re.Match[str]) -> bool:
    prefix = field_value[: match.start()].rstrip()
    anchor_len = _anchor_length(prefix)
    if not anchor_len:
        return False
    # Alumni stay in the pool because the exclusion keys on current only.
    return not _names_past_employer(prefix[: len(prefix) - anchor_len])


def _text_names_employer_any_tense(field_value: str, match: re.Match[str]) -> bool:
    """Whether free text names the match as an employer, current or past."""

    prefix = field_value[: match.start()]
    if _anchor_length(prefix.rstrip()):
        return True
    if _PAST_MARKER_BEFORE_NAME.search(prefix):
        return True
    return _ALUMNI_WORD_AFTER_NAME.match(field_value[match.end():]) is not None


def _names_past_employer(before_anchor: str) -> bool:
    """Whether a past marker governs the anchor that ``before_anchor`` precedes.

    The marker counts when it sits in the anchor's own stretch of text: the
    clause the anchor is in, cut after any "now" / "currently" or earlier
    anchor. Inside that stretch it counts within the lookback window before
    the anchor, or when it opens the stretch.
    """

    clause_start = 0
    for separator in _CLAUSE_SEPARATOR.finditer(before_anchor):
        clause_start = separator.end()
    stretch = before_anchor[clause_start:]
    stretch_start = 0
    for stop in _PAST_STRETCH_END.finditer(stretch):
        stretch_start = stop.end()
    stretch = stretch[stretch_start:]
    # Search the whole stretch so a cut word ("Transformer" -> "former") can
    # never pass for a marker, then keep markers inside the window.
    window_start = len(stretch) - _LOOKBACK_CHARS
    for marker in _PAST_EMPLOYER_MARKER.finditer(stretch):
        if marker.start() >= window_start:
            return True
    return _PAST_EMPLOYER_MARKER.match(stretch.lstrip(" \t:-([")) is not None


__all__ = ["match_any_employer", "match_current_employer"]
