"""Town-level ``excluded_locations`` entries for whole counties, from the Census.

A brief's ``excluded_locations`` matches a search card's location by whole
comma-separated parts (shared/excluded_locations.py), and LinkedIn usually shows
the person's own town, so excluding a county means listing every town in it.
Hand-picked lists leak: the SPL SWE v2 list named 238 entries and missed
Syosset, Hartsdale, Livingston, Paramus and 128 other historical cards inside
the counties it claimed to exclude.

This reads two Census 2020 reference files and prints, for the named counties,
every incorporated place, census-designated place, and (for states with
governmental towns and townships, such as New York and New Jersey) county
subdivision, written "Town, State" with the legal suffix dropped
("Livingston township" -> "Livingston, New Jersey").

A name the same state also uses OUTSIDE the named counties is a collision
("Harrison" is a Hudson County town and a Gloucester County township). A card
cannot say which one it means, so collisions are printed to stderr and left
out unless named with --include; an excluded card is a person no one sources,
which costs more than an occasional overlap with another recruiter.

Usage:
    .venv/bin/python tools/excluded_locations_from_census.py \\
        --county "NJ:Hudson" --county "NJ:Bergen" --include "NJ:Harrison" \\
        [--place-file national_place_by_county2020.txt] \\
        [--cousub-file national_cousub2020.txt]

Without the file flags it downloads both from www2.census.gov.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

PLACE_URL = (
    "https://www2.census.gov/geo/docs/reference/codes2020/"
    "national_place_by_county2020.txt"
)
COUSUB_URL = (
    "https://www2.census.gov/geo/docs/reference/codes2020/national_cousub2020.txt"
)

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

# County subdivisions that are a government people name as home. CCDs (Z5)
# are statistical only; the rest of the West has no governmental subdivisions.
COUSUB_STATES = frozenset({"NY", "NJ", "PA", "CT", "MA", "RI", "NH", "VT", "ME", "MI", "WI", "MN"})
_STATISTICAL_COUSUB_CLASSES = frozenset({"Z1", "Z5", "Z7", "Z9"})

_LEGAL_SUFFIX = re.compile(
    r"\s+(city and borough|consolidated government \(balance\)|city|town|village|"
    r"CDP|borough|township|charter township|municipality|plantation|"
    r"unified government \(balance\)|metro government \(balance\))$",
    re.IGNORECASE,
)


def town_name(census_name: str) -> str:
    """'Livingston township' -> 'Livingston'; '(balance)' and parentheses dropped."""

    name = re.sub(r"\s*\([^)]*\)", "", census_name).strip()
    return _LEGAL_SUFFIX.sub("", name).strip()


def _rows(text: str) -> list[list[str]]:
    lines = text.splitlines()
    return [line.split("|") for line in lines[1:] if line.strip()]


def county_towns(
    place_text: str,
    cousub_text: str,
    counties: list[tuple[str, str]],
) -> tuple[list[str], dict[str, list[str]]]:
    """``(entries, collisions)`` for the named ``(state, county)`` pairs.

    ``entries`` are "Town, State" names found only inside the named counties
    of their state; ``collisions`` maps each other name to the outside
    counties that also carry it.
    """

    wanted = {(state, county.removesuffix(" County")) for state, county in counties}
    states = {state for state, _ in wanted}
    # (state, casefolded name) -> {(kind, fips id, county)}
    inside: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    outside: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    display: dict[tuple[str, str], str] = {}

    def add(kind: str, state: str, county: str, fips: str, census_name: str) -> None:
        name = town_name(census_name)
        if not name:
            return
        key = (state, name.casefold())
        display.setdefault(key, name)
        bucket = inside if (state, county) in wanted else outside
        bucket[key].add((kind, fips, county))

    for row in _rows(place_text):
        state, county, fips, census_name = row[0], row[3], row[4], row[6]
        if state in states:
            add("place", state, county.removesuffix(" County"), fips, census_name)
    for row in _rows(cousub_text):
        state, county, fips, census_name, cls = row[0], row[3], row[4], row[6], row[7]
        if state not in states or state not in COUSUB_STATES:
            continue
        if cls in _STATISTICAL_COUSUB_CLASSES or "not defined" in census_name:
            continue
        add("cousub", state, county.removesuffix(" County"), fips, census_name)

    entries: list[str] = []
    collisions: dict[str, list[str]] = {}
    for key in sorted(inside, key=lambda k: (k[0], display[k])):
        state = key[0]
        label = f"{display[key]}, {STATE_NAMES[state]}"
        same_feature = {(kind, fips) for kind, fips, _ in inside[key]}
        elsewhere = sorted(
            {county for kind, fips, county in outside.get(key, set())
             if (kind, fips) not in same_feature}
        )
        if elsewhere:
            collisions[label] = elsewhere
        else:
            entries.append(label)
    return entries, collisions


def _county_arg(raw: str) -> tuple[str, str]:
    state, sep, county = raw.partition(":")
    if not sep or state.upper() not in STATE_NAMES or not county.strip():
        raise argparse.ArgumentTypeError(f"--county wants ST:County, got {raw!r}")
    return state.upper(), county.strip()


def _read(path: str | None, url: str) -> str:
    if path:
        return Path(path).read_text(encoding="latin-1")
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("latin-1")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--county", type=_county_arg, action="append", required=True)
    ap.add_argument("--include", action="append", default=[],
                    help="ST:Town — keep this collision (the named county's town dominates)")
    ap.add_argument("--place-file")
    ap.add_argument("--cousub-file")
    args = ap.parse_args(argv)

    entries, collisions = county_towns(
        _read(args.place_file, PLACE_URL),
        _read(args.cousub_file, COUSUB_URL),
        args.county,
    )
    included = set()
    for raw in args.include:
        state, _, town = raw.partition(":")
        label = f"{town.strip()}, {STATE_NAMES[state.upper()]}"
        if label not in collisions:
            ap.error(f"--include {raw!r} is not a collision in the named counties")
        included.add(label)
    for label, elsewhere in collisions.items():
        verdict = "KEPT (--include)" if label in included else "left out"
        print(f"collision: {label} also in {', '.join(elsewhere)} — {verdict}", file=sys.stderr)
    print(json.dumps(sorted(set(entries) | included), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
