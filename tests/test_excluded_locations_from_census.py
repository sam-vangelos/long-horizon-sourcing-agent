"""``tools/excluded_locations_from_census.py``: whole counties as town entries."""

from __future__ import annotations

import json

import tools.excluded_locations_from_census as census

PLACES = """STATE|STATEFP|COUNTYFP|COUNTYNAME|PLACEFP|PLACENS|PLACENAME|TYPE|CLASSFP|FUNCSTAT
NJ|34|003|Bergen County|55950|00885341|Paramus borough|INCORPORATED PLACE|C1|A
NJ|34|013|Essex County|67080|02389869|Short Hills CDP|CENSUS DESIGNATED PLACE|U1|S
NJ|34|017|Hudson County|30210|00885245|Harrison town|INCORPORATED PLACE|C1|A
NJ|34|021|Mercer County|60900|00885355|Princeton|INCORPORATED PLACE|C1|A
NY|36|059|Nassau County|72554|02389976|Syosset CDP|CENSUS DESIGNATED PLACE|U1|S
CA|06|085|Santa Clara County|09066|02583176|Burbank CDP|CENSUS DESIGNATED PLACE|U1|S
CA|06|037|Los Angeles County|08954|02409939|Burbank city|INCORPORATED PLACE|C1|A
"""

COUSUBS = """STATE|STATEFP|COUNTYFP|COUNTYNAME|COUSUBFP|COUSUBNS|COUSUBNAME|CLASSFP|FUNCSTAT
NJ|34|013|Essex County|41310|00882226|Livingston township|T1|A
NJ|34|015|Gloucester County|30180|00882107|Harrison township|T1|A
NJ|34|003|Bergen County|55950|00885341|Paramus borough|T5|A
NY|36|005|Bronx County|08510|00978756|Bronx borough|Z1|G
CA|06|085|Santa Clara County|92290|01935107|San Jose CCD|Z5|S
"""


def test_towns_and_townships_inside_the_counties_become_entries() -> None:
    entries, collisions = census.county_towns(
        PLACES,
        COUSUBS,
        [("NJ", "Bergen"), ("NJ", "Essex"), ("NJ", "Hudson"), ("NY", "Nassau")],
    )

    assert entries == [
        "Livingston, New Jersey",
        "Paramus, New Jersey",
        "Short Hills, New Jersey",
        "Syosset, New York",
    ]
    # Hudson's Harrison town shares its name with a Gloucester township.
    assert collisions == {"Harrison, New Jersey": ["Gloucester"]}


def test_a_name_another_county_also_uses_is_a_collision_not_an_entry() -> None:
    entries, collisions = census.county_towns(PLACES, COUSUBS, [("CA", "Santa Clara")])

    assert entries == []
    assert collisions == {"Burbank, California": ["Los Angeles"]}


def test_statistical_subdivisions_never_become_entries() -> None:
    entries, _ = census.county_towns(PLACES, COUSUBS, [("NY", "Bronx"), ("CA", "Santa Clara")])

    assert "Bronx, New York" not in entries
    assert "San Jose, California" not in entries


def test_cli_keeps_only_the_collisions_it_is_told_to(tmp_path, capsys) -> None:
    place_file = tmp_path / "places.txt"
    cousub_file = tmp_path / "cousubs.txt"
    place_file.write_text(PLACES)
    cousub_file.write_text(COUSUBS)

    assert census.main([
        "--place-file", str(place_file), "--cousub-file", str(cousub_file),
        "--county", "NJ:Hudson", "--include", "NJ:Harrison",
    ]) == 0

    out = capsys.readouterr()
    assert json.loads(out.out) == ["Harrison, New Jersey"]
    assert "collision: Harrison, New Jersey also in Gloucester — KEPT (--include)" in out.err


def test_town_name_drops_the_legal_suffix() -> None:
    assert census.town_name("Livingston township") == "Livingston"
    assert census.town_name("St. Helena city") == "St. Helena"
    assert census.town_name("Syosset CDP") == "Syosset"
