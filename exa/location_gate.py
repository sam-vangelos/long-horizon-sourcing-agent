"""Location scope for Exa briefs that source in named countries only.

The judge reads location as prompt text and, on the SPL competitor run, misread
it: a card that says Edmonton read as "a United Kingdom tag", and a company's
headquarters line on a role ("Stanford, CA" on every Pareto seat) read as the
person's home when the card said Nairobi. This module is the deterministic
check in front of the judge. It reads the person's CARD location and nothing
else, and excludes only a location that resolves POSITIVELY to a country
outside the brief's list. An empty or unresolvable location goes to the judge.

The check is inert unless the brief opts in with the top-level block
``exa_location_scope``::

    {"countries": ["United States"]}

Entries are country names or ISO 3166-1 alpha-2 codes; a name this module
cannot resolve is a blocking lint finding, because a scope it cannot read would
either exclude everyone or no one. The United States includes its territories.

Resolution, in order:

1. The LinkedIn render's location line. Exa appends the country code
   ("Hyderabad, Telangana, India (IN)"), and that code decides.
2. The same line without a code, or the record's own ``location`` when there
   is no render: the last comma part, or the whole label, must be a country
   name or a US state. A label that is both (Georgia) resolves to nothing.
3. Anything else ("Greater Seattle Area", "Bay Area", "Brookline, MA") is
   UNKNOWN and passes.

A role's own location line is never read: it is where the company put the
seat, often its headquarters, not where the person lives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from shared.schemas import OpusDecision

from exa.linkedin_render import strip_country_suffix
from exa.schemas import ExaCandidate


LOCATION_FIELD = "exa_location_scope"
COUNTRIES_KEY = "countries"
LOCATION_SCOPE_KEYS = (COUNTRIES_KEY,)

LOCATION_GATE_PATH = "location_scope"
LOCATION_GATE_REJECT_REASON = "HARD_GATE"

INSIDE = "inside"
OUTSIDE = "outside"
UNKNOWN = "unknown"

# Country names (lowercased) to ISO 3166-1 alpha-2. Covers every country Exa's
# card codes have shown on the stored campaigns plus the usual sourcing
# geographies; a code the table lacks still resolves through rule 1.
_COUNTRY_NAME_TO_CODE: dict[str, str] = {
    "united states": "US", "united states of america": "US", "usa": "US",
    "us": "US", "u.s.": "US", "u.s.a.": "US", "america": "US",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "england": "GB",
    "scotland": "GB", "wales": "GB", "northern ireland": "GB",
    "canada": "CA", "mexico": "MX", "brazil": "BR", "argentina": "AR",
    "chile": "CL", "colombia": "CO", "peru": "PE", "uruguay": "UY",
    "venezuela": "VE", "ecuador": "EC", "costa rica": "CR",
    "ireland": "IE", "france": "FR", "germany": "DE", "spain": "ES",
    "portugal": "PT", "italy": "IT", "netherlands": "NL",
    "the netherlands": "NL", "belgium": "BE", "switzerland": "CH",
    "austria": "AT", "sweden": "SE", "norway": "NO", "denmark": "DK",
    "finland": "FI", "poland": "PL", "czechia": "CZ", "czech republic": "CZ",
    "romania": "RO", "ukraine": "UA", "greece": "GR", "turkey": "TR",
    "türkiye": "TR", "turkiye": "TR", "israel": "IL",
    "united arab emirates": "AE", "uae": "AE", "india": "IN",
    "pakistan": "PK", "bangladesh": "BD", "sri lanka": "LK", "nepal": "NP",
    "china": "CN", "hong kong": "HK", "taiwan": "TW", "japan": "JP",
    "south korea": "KR", "korea": "KR", "singapore": "SG", "malaysia": "MY",
    "indonesia": "ID", "thailand": "TH", "vietnam": "VN", "viet nam": "VN",
    "philippines": "PH", "australia": "AU", "new zealand": "NZ",
    "south africa": "ZA", "nigeria": "NG", "kenya": "KE", "ghana": "GH",
    "uganda": "UG", "ethiopia": "ET", "egypt": "EG", "morocco": "MA",
    "estonia": "EE", "latvia": "LV", "lithuania": "LT", "hungary": "HU",
    "bulgaria": "BG", "serbia": "RS", "croatia": "HR", "slovakia": "SK",
    "slovenia": "SI", "iceland": "IS", "luxembourg": "LU", "russia": "RU",
    "armenia": "AM", "georgia": "GE", "jordan": "JO", "lebanon": "LB",
    "saudi arabia": "SA", "qatar": "QA", "puerto rico": "PR",
}

# A country's own territories count as the country. Only the United States
# needs this today: Exa codes Puerto Rico as PR.
_TERRITORY_PARENT: dict[str, str] = {
    "PR": "US", "GU": "US", "VI": "US", "AS": "US", "MP": "US", "UM": "US",
}

# US states as LinkedIn writes them on a card. Georgia is deliberately absent:
# it is also a country, and a bare "Tbilisi, Georgia" must not read as US.
_US_STATE_NAMES = frozenset(
    (
        "alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
        "delaware|florida|hawaii|idaho|illinois|indiana|iowa|kansas|"
        "kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
        "mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
        "new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|"
        "pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|"
        "utah|vermont|virginia|washington|west virginia|wisconsin|wyoming|"
        "district of columbia|washington dc|washington d.c."
    ).split("|")
)

# Labels that name a place in more than one country; they resolve to nothing.
_AMBIGUOUS_LABELS = frozenset({"georgia"})

# Exa's trailing country code on a card location: "... (IN)".
_CODE_SUFFIX_RE = re.compile(r"\(([A-Z]{2})\)\s*$")

# A display name per code, for the console and the rationale.
_CODE_DISPLAY: dict[str, str] = {
    "US": "the United States",
    "GB": "the United Kingdom",
    "AE": "the United Arab Emirates",
    "NL": "the Netherlands",
    "PH": "the Philippines",
}


def _flatten(value: object) -> str:
    return " ".join(str(value or "").split())


def _code_for_name(name: str) -> str | None:
    """ISO alpha-2 for a country name or an alpha-2 code; None if unknown."""

    text = _flatten(name)
    if not text:
        return None
    if len(text) == 2 and text.isalpha() and text.isupper():
        return text
    return _COUNTRY_NAME_TO_CODE.get(text.casefold())


def _country_display(code: str) -> str:
    if code in _CODE_DISPLAY:
        return _CODE_DISPLAY[code]
    for name, value in _COUNTRY_NAME_TO_CODE.items():
        if value == code and len(name) > 3:
            return name.title()
    return code


def _parent(code: str) -> str:
    return _TERRITORY_PARENT.get(code, code)


def _scope_block(brief: Any) -> Any:
    """The block off a raw dict or either loaded brief shape; None when absent."""

    if isinstance(brief, Mapping):
        return brief.get(LOCATION_FIELD)
    raw = getattr(brief, "raw", None)
    if isinstance(raw, Mapping):
        return raw.get(LOCATION_FIELD)
    return getattr(brief, LOCATION_FIELD, None)


def location_scope_errors(brief: Any) -> list[str]:
    """What is wrong with the brief's location block, as short phrases."""

    block = _scope_block(brief)
    if block is None:
        return []
    if not isinstance(block, dict):
        return [f"{LOCATION_FIELD} must be an object"]
    errors = [f"unknown key {key!r}" for key in block if key not in LOCATION_SCOPE_KEYS]
    countries = block.get(COUNTRIES_KEY)
    if not isinstance(countries, list) or not countries:
        errors.append(f"{COUNTRIES_KEY} must be a non-empty list of country names")
        return errors
    for entry in countries:
        if not isinstance(entry, str) or _code_for_name(entry) is None:
            errors.append(
                f"country {entry!r} is not a country name or two-letter code this "
                "check can resolve"
            )
    return errors


@dataclass(frozen=True)
class LocationVerdict:
    """Where the record places the person, against the scope."""

    status: str
    location: str = ""
    country: str = ""

    def note(self, scope: LocationScope) -> str:
        if self.status != OUTSIDE:
            return ""
        return f"outside {scope.label} ({self.location})"


@dataclass(frozen=True)
class LocationScope:
    """The countries a candidate's card location must resolve inside."""

    codes: frozenset[str]
    names: tuple[str, ...]

    @classmethod
    def from_brief(cls, brief: Any) -> LocationScope | None:
        """The scope, or None when the brief does not opt in.

        Raises ``ValueError`` on a malformed block: the lint refuses such a
        brief before any spend, so a run never starts without the check.
        """

        block = _scope_block(brief)
        if block is None:
            return None
        errors = location_scope_errors(brief)
        if errors:
            raise ValueError(f"{LOCATION_FIELD} in the brief is malformed: " + "; ".join(errors))
        names = tuple(_flatten(entry) for entry in block[COUNTRIES_KEY])
        codes = frozenset(_parent(_code_for_name(name) or "") for name in names)
        return cls(codes=codes, names=names)

    @property
    def label(self) -> str:
        """"the United States", or "the United States and Canada"."""

        shown = [_country_display(code) for code in sorted(self.codes)]
        if len(shown) == 1:
            return shown[0]
        return ", ".join(shown[:-1]) + f" and {shown[-1]}"

    def check(self, candidate: ExaCandidate) -> LocationVerdict:
        """INSIDE or OUTSIDE on a resolved card location; UNKNOWN otherwise."""

        location, code = _card_country(candidate)
        if code is None:
            return LocationVerdict(UNKNOWN, location=location)
        status = INSIDE if _parent(code) in self.codes else OUTSIDE
        return LocationVerdict(status, location=location, country=code)


def _card_country(candidate: ExaCandidate) -> tuple[str, str | None]:
    """``(location as shown, country code or None)`` off the person's card."""

    render = candidate.render()
    card = _flatten(render.location) if render is not None else ""
    if card:
        match = _CODE_SUFFIX_RE.search(card)
        shown = strip_country_suffix(card)
        if match:
            return shown, match.group(1)
        return shown, _country_from_text(shown)
    shown = _flatten(candidate.location)
    return shown, _country_from_text(shown)


def _country_from_text(text: str) -> str | None:
    """A country code from a card label with no code; None unless unambiguous."""

    parts = [part.strip().casefold() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    last = parts[-1]
    if last in _AMBIGUOUS_LABELS:
        return None
    if last in _COUNTRY_NAME_TO_CODE:
        return _COUNTRY_NAME_TO_CODE[last]
    if last in _US_STATE_NAMES:
        return "US"
    return None


def location_gate_decision(
    candidate: ExaCandidate, verdict: LocationVerdict, scope: LocationScope
) -> OpusDecision:
    """The deterministic REJECT a candidate outside the scope is stored under."""

    return OpusDecision(
        stage="full",
        decision="REJECT",
        path=LOCATION_GATE_PATH,
        confidence=1.0,
        rationale=f"located {verdict.note(scope)}",
        candidate_name=candidate.name,
        profile_url=candidate.linkedin_url or (candidate.evidence_urls() or [""])[0],
        reject_reason=LOCATION_GATE_REJECT_REASON,
    )


def is_location_gate_decision(decision: object) -> bool:
    return getattr(decision, "path", "") == LOCATION_GATE_PATH


__all__ = [
    "COUNTRIES_KEY",
    "INSIDE",
    "LOCATION_FIELD",
    "LOCATION_GATE_PATH",
    "OUTSIDE",
    "UNKNOWN",
    "LocationScope",
    "LocationVerdict",
    "is_location_gate_decision",
    "location_gate_decision",
    "location_scope_errors",
]
