"""Deterministic location exclusion for a brief's ``excluded_locations``.

LinkedIn Recruiter's Location filter can only include, so a territory such as
"the United States except four markets" keeps the filter broad and excludes
here, on the location the search card shows. An entry matches when its
comma-separated parts appear, in order and whole, in the location's parts:
"New York, New York" matches "New York, New York, United States" and never
"Buffalo, New York, United States". Anything the list does not name is in
scope, so an unresolvable location is never excluded.

A card can carry LinkedIn's industry after the place ("San Francisco Bay
Area · Legal Services"). The model extractor keeps that suffix, so the
location is cut at the first " · " before matching, the rule the
deterministic card extractor applies (shared/extractors.py).
"""

from __future__ import annotations

from typing import Iterable

# The facial gate path a territory skip records on its canonical stage row
# (linkedin/candidate_review.py). Defined here so the ledger can read it back
# on resume without importing the reviewer.
TERRITORY_GATE_PATH = "excluded_location"


_INDUSTRY_SEPARATOR = " · "


def _place(location: str) -> str:
    """The place part of a card location, without LinkedIn's " · Industry" suffix."""

    return location.split(_INDUSTRY_SEPARATOR, 1)[0]


def _parts(text: str) -> list[str]:
    return [part.strip().casefold() for part in text.split(",") if part.strip()]


def excluded_location_match(
    location: str | None,
    excluded_locations: Iterable[str] | None,
) -> str | None:
    """The first entry of ``excluded_locations`` that ``location`` falls in, else None."""

    if not location or not excluded_locations:
        return None
    location_parts = _parts(_place(location))
    for entry in excluded_locations:
        entry_parts = _parts(str(entry or ""))
        width = len(entry_parts)
        if not width:
            continue
        for start in range(len(location_parts) - width + 1):
            if location_parts[start:start + width] == entry_parts:
                return str(entry).strip()
    return None
