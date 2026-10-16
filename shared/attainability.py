"""Attainability: would this person plausibly leave their current seat?

The judge scores fit to the role. It never scores whether a person would
consider leaving what they hold now, so a slate can lead with people who will
not take the call. This rule answers that from the stored record, with no
model call: a candidate is a ``long_shot`` when their CURRENT employer is on
the brief's ``long_shot_employers`` list or their CURRENT title matches one of
its ``long_shot_title_patterns``, and ``reachable`` otherwise. Slates sort
long shots last and say why; nobody is removed.

Employer matching is :func:`shared.employer_blacklist.match_current_employer`,
the rule the blacklist gates use: a company field matches on a word boundary.
A headline or title is read only when the record names no current company,
and then only where it says "at X" or "@ X": headlines mix present and past
("Product Lead @ Mercor || Ex-SWE @ Meta"), and the recorded company is the
better witness. Title patterns are case-insensitive regular expressions, read
against the title only: the whole title and each of its segments (split on
",", "|", "&", "/", "·", ";", "(", " - ", " and "), so a pattern anchored with
"^" matches at the start of the title or of a segment. The defaults are
written that way, which keeps "Co-Founder & CTO" a long shot and "Chief of
Staff to the CEO" or "Founder's Associate" reachable.

The brief's ``attainability`` block supplies the inputs. A key the brief
leaves out takes its default from ``attainability_defaults.json``; a key set
to ``[]`` switches that half of the rule off.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from shared.employer_blacklist import match_current_employer

LONG_SHOT = "long_shot"
REACHABLE = "reachable"

EMPLOYERS_KEY = "long_shot_employers"
TITLE_PATTERNS_KEY = "long_shot_title_patterns"
ATTAINABILITY_KEYS = (EMPLOYERS_KEY, TITLE_PATTERNS_KEY)

# Brief-shaped data rather than code: the same block a brief would carry.
_DEFAULTS_PATH = Path(__file__).with_name("attainability_defaults.json")


def _load_defaults() -> dict[str, tuple[str, ...]]:
    block = json.loads(_DEFAULTS_PATH.read_text(encoding="utf-8"))
    errors = attainability_errors(block)
    if errors:
        raise ValueError(f"{_DEFAULTS_PATH.name} is malformed: {errors}")
    return {key: tuple(block[key]) for key in ATTAINABILITY_KEYS}


def attainability_errors(block: Any) -> list[str]:
    """What is wrong with an ``attainability`` block, as short phrases.

    Empty when the block is absent (``None``) or well formed. Unknown keys
    are errors: a misspelt key would otherwise leave the default in force
    with nothing to say so.
    """

    if block is None:
        return []
    if not isinstance(block, dict):
        return ["attainability must be an object"]
    errors = [f"unknown key {key!r}" for key in block if key not in ATTAINABILITY_KEYS]
    for key in ATTAINABILITY_KEYS:
        if key not in block:
            continue
        values = block[key]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            errors.append(f"{key} must be a list of non-empty strings")
            continue
        if key == TITLE_PATTERNS_KEY:
            for pattern in values:
                try:
                    re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    errors.append(f"{key} entry {pattern!r} is not a valid pattern: {exc}")
    return errors


_DEFAULTS = _load_defaults()
DEFAULT_LONG_SHOT_EMPLOYERS: tuple[str, ...] = _DEFAULTS[EMPLOYERS_KEY]
DEFAULT_LONG_SHOT_TITLE_PATTERNS: tuple[str, ...] = _DEFAULTS[TITLE_PATTERNS_KEY]


@dataclass(frozen=True)
class AttainabilityRules:
    long_shot_employers: tuple[str, ...] = DEFAULT_LONG_SHOT_EMPLOYERS
    long_shot_title_patterns: tuple[str, ...] = DEFAULT_LONG_SHOT_TITLE_PATTERNS

    @classmethod
    def from_brief(cls, brief: Any) -> AttainabilityRules:
        """Rules from a brief: a raw JSON dict or a loaded brief.

        A loaded brief is either the V2 dataclass, which carries the block
        as ``attainability``, or what ``load_brief`` returns, which carries
        the JSON it was loaded from as ``raw`` (V1 seeds included). Raises
        ``ValueError`` on a malformed block rather than falling back, so a
        typo in the brief never passes for the defaults.
        """

        if isinstance(brief, Mapping):
            block = brief.get("attainability")
        elif hasattr(brief, "attainability"):
            block = brief.attainability
        else:
            block = dict(getattr(brief, "raw", None) or {}).get("attainability")
        errors = attainability_errors(block)
        if errors:
            raise ValueError("attainability in the brief is malformed: " + "; ".join(errors))
        block = block or {}
        return cls(
            long_shot_employers=tuple(
                value.strip() for value in block.get(EMPLOYERS_KEY, DEFAULT_LONG_SHOT_EMPLOYERS)
            ),
            long_shot_title_patterns=tuple(
                block.get(TITLE_PATTERNS_KEY, DEFAULT_LONG_SHOT_TITLE_PATTERNS)
            ),
        )


def _texts(values: Iterable[str | None]) -> list[str]:
    return [" ".join(str(value).split()) for value in values if value and str(value).strip()]


# Where one role in a title ends and the next begins: "Co-Founder & CTO",
# "Co-founder, Head of Sales", "CTO / Co-Founder", "Founder and CEO".
_TITLE_SEGMENT_SEPARATOR = re.compile(
    r"\s*[,|&/·;(]\s*|\s+[-\u2013\u2014]\s+|\s+and\s+", re.IGNORECASE
)


def _title_readings(title: str) -> list[str]:
    """The title, then each segment of it, for the title patterns to read."""

    readings = [title]
    for segment in _TITLE_SEGMENT_SEPARATOR.split(title):
        segment = segment.strip(" )")
        if segment and segment not in readings:
            readings.append(segment)
    return readings


def classify_signals(
    *,
    companies: Iterable[str | None],
    headlines: Iterable[str | None],
    titles: Iterable[str | None],
    rules: AttainabilityRules,
) -> tuple[str, str]:
    """``(tier, reason)`` from every current-seat signal a record carries.

    ``companies`` match on a word boundary; ``headlines`` (which should
    include the titles) are read only when ``companies`` is empty, and match
    only in "at X" / "@ X" context; ``titles`` are matched against the title
    patterns, whole and segment by segment. An employer match is reported
    ahead of a title match.
    """

    employers = list(rules.long_shot_employers)
    company_texts = _texts(companies)
    headline_texts = [] if company_texts else _texts(headlines)
    if employers:
        for company in company_texts:
            hit = match_current_employer(company, "", employers)
            if hit is not None:
                return LONG_SHOT, f"current employer: {hit[0]}"
        for headline in headline_texts:
            hit = match_current_employer("", headline, employers)
            if hit is not None:
                return LONG_SHOT, f"current employer: {hit[0]}"
    title_readings = [_title_readings(title) for title in _texts(titles)]
    for pattern in rules.long_shot_title_patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for readings in title_readings:
            for reading in readings:
                match = compiled.search(reading)
                if match is not None:
                    return LONG_SHOT, f"title: {match.group(0)}"
    return REACHABLE, ""


def classify_attainability(
    current_company: str | None,
    title: str | None,
    rules: AttainabilityRules,
    *,
    headline: str | None = None,
) -> tuple[str, str]:
    """``("long_shot" | "reachable", reason)`` for one current seat.

    ``reason`` is the matched value ("current employer: OpenAI",
    "title: Co-Founder"), or empty for a reachable candidate.
    """

    return classify_signals(
        companies=[current_company],
        headlines=[title, headline],
        titles=[title],
        rules=rules,
    )


__all__ = [
    "ATTAINABILITY_KEYS",
    "DEFAULT_LONG_SHOT_EMPLOYERS",
    "DEFAULT_LONG_SHOT_TITLE_PATTERNS",
    "LONG_SHOT",
    "REACHABLE",
    "AttainabilityRules",
    "attainability_errors",
    "classify_attainability",
    "classify_signals",
]
