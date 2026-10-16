"""Bound a campaign to named CURRENT employers: a brief's ``employer_allowlist``.

A competitor-sourcing campaign sources only people
currently employed at the hiring company's competitors in AI training data and
evaluation. LinkedIn Recruiter's company facet matches current OR past
employees, and adaptation, variant, and typeahead-miss paths can each run a
string with no competitor facet, so the restriction is enforced here, with no
model call:

- the LinkedIn facial gate and post-profile gate skip a person whose current
  employer is positively identified and is not a listed employer
  (:func:`current_employer_scope`);
- the per-string facet stamp puts a listed group of employers on every string
  that lacks one (``linkedin/competitor_scope.py``).

The seed field::

    "employer_allowlist": {
      "facet_groups": [["Mercor"], ["Handshake"], ["Scale AI"], ["Appen", "Sama"]],
      "aliases": {"Outlier": "Scale AI", "Handshake AI": "Handshake"}
    }

``facet_groups`` are the listed employers, in the groups a company facet
carries them (at most :data:`FACET_COMPANY_VALUE_CAP` per group). ``aliases``
map other names (contractor brands, a renamed company) to a listed employer
for SCOPE only; an alias is never typed into a facet. Keys starting with "_"
are operator notes. A brief without the field, or with a malformed one, is
untouched by every rule here.

The optional ``title_restricted`` block bounds listed employers that count
only in some seats (for req 1238, the AI-data vendors' clients, in scope on
their human-data, post-training and evaluation side alone)::

    "title_restricted": {
      "employers": ["Meta", "Amazon"],
      "facet_titles": ["Human Data Manager", "Post-training Program Manager"],
      "title_markers": ["human data", "post training", "evaluation"],
      "title_exclusions": ["engineer", "scientist", "director"]
    }

A title is in the families when it names a marker and every exclusion word it
carries also sits inside a marker it names ("Human Data Engineer" passes on
the marker "human data engineer"; "Software Engineer, Evaluations" fails).
A restricted employer keeps a person in scope only when a current title or
the headline is in the families. Its facet group may hold no other employer,
and every facet title must itself be in the families. The facet stamp, the
string skip, and the variant rules read the same block
(``linkedin/competitor_scope.py``, ``linkedin/search_policy.py``).

Matching is generous toward "in scope", because a false out-of-scope skip
settles a person as FACIAL_NO for good while a false in-scope costs one judge
call: punctuation and spacing fold away ("Pareto.AI", "Data Annotation"), a
LinkedIn employment-type suffix is dropped ("AfterQuery Experts · Contract"),
and a truncated company name counts when the listed name only adds a generic
suffix ("Invisible" for "Invisible Technologies"). A current employer the
record does not name, or names only as "Self-employed", "Stealth", or
"Startup", is not identified, and the person passes to the judge.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from shared.employer_blacklist import match_current_employer

EMPLOYER_ALLOWLIST_KEY = "employer_allowlist"

# The decision path the LinkedIn gates stamp on an out-of-scope skip. It lives
# here, beside TERRITORY_GATE_PATH's home in shared/excluded_locations.py, so
# the resume rebuild (linkedin/candidate_ledger.py) can read it back without
# importing the reviewer.
EMPLOYER_SCOPE_GATE_PATH = "employer_scope"

# The most company values one string's facet carries. No code path in
# linkedin/ limits a facet's value count, and LinkedIn documents none; 12 is
# the largest company facet a live receipt shows landing whole (the PRRE
# Brazil run log: 12 desired, 12 after, 0 unapplied). Every value is one
# typeahead round trip, so the cap also bounds the opening's cost.
FACET_COMPANY_VALUE_CAP = 12

# Words a listed name may add to a truncated company and still name it:
# "Invisible" is "Invisible Technologies", "Surge" is "Surge AI". "Uber" is
# never "Uber AI Solutions", because "solutions" is not one of them.
_GENERIC_SUFFIX_MARKERS = frozenset(
    {"ai", "technologies", "technology", "tech", "labs", "lab", "inc", "io", "co", "group", "data"}
)

# The keys a ``title_restricted`` block may carry ("_" keys are notes).
_TITLE_RESTRICTED_KEYS = frozenset(
    {"employers", "facet_titles", "title_markers", "title_exclusions"}
)

# A truncated company must carry at least this many letters to count.
_MIN_TRUNCATED_CHARS = 4

# Company fields that do not identify an employer.
_PLACEHOLDER_COMPANY_MARKERS = frozenset(
    {
        "self employed",
        "selfemployed",
        "freelance",
        "freelancer",
        "freelancing",
        "independent",
        "independent contractor",
        "independent consultant",
        "contract",
        "contractor",
        "consultant",
        "consulting",
        "confidential",
        # LinkedIn's placeholder companies for an employer the person does
        # not name. Only the whole field counts: "Startup Grind" is a company.
        "startup",
        "a startup",
        "early stage startup",
        "ai startup",
        "stealth",
        "stealth company",
        "stealth startup",
        "stealth mode",
        "stealth mode startup",
        "stealth ai startup",
        "various",
        "n a",
        "na",
        "none",
        "unemployed",
        "career break",
        "open to work",
    }
)

_PRESENT_TAIL = re.compile(r"\s*[\(·]\s*[^()·]*\bpresent\b[^()·]*\)?\s*$", re.IGNORECASE)


def _fold_tokens(text: object) -> list[str]:
    """Casefolded alphanumeric tokens with accents stripped."""

    raw = unicodedata.normalize("NFKD", str(text or ""))
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", " ", raw.casefold()).split()


def _strip_employment_suffix(company: object) -> str:
    """'AfterQuery Experts · Contract' -> 'AfterQuery Experts'."""

    return str(company or "").split("·", 1)[0].strip()


def _contains_window(haystack: list[str], needle: list[str]) -> bool:
    """Whether ``needle`` sits in ``haystack`` as whole tokens, spacing folded.

    "data annotation" and "dataannotation" are one name, so each contiguous
    window of the haystack is compared with its spaces removed as well.
    """

    if not needle or not haystack:
        return False
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        if haystack[start:start + width] == needle:
            return True
    squashed = "".join(needle)
    for start in range(len(haystack)):
        joined = ""
        for end in range(start, len(haystack)):
            joined += haystack[end]
            if joined == squashed:
                return True
            if len(joined) >= len(squashed):
                break
    return False


# Trailing words a LinkedIn entity name may carry beyond the listed name.
_CORPORATE_SUFFIX_MARKERS = _GENERIC_SUFFIX_MARKERS | frozenset(
    {"llc", "ltd", "corp", "corporation", "company", "incorporated", "limited", "gmbh", "plc"}
)


def _core_tokens(tokens: list[str]) -> list[str]:
    """Tokens with trailing corporate suffixes removed (never down to nothing)."""

    core = list(tokens)
    while len(core) > 1 and core[-1] in _CORPORATE_SUFFIX_MARKERS:
        core.pop()
    return core


def _is_truncation(company_tokens: list[str], name_tokens: list[str]) -> bool:
    """Whether a company is the listed name minus a generic suffix."""

    if not company_tokens or len(company_tokens) >= len(name_tokens):
        return False
    if name_tokens[: len(company_tokens)] != company_tokens:
        return False
    if len("".join(company_tokens)) < _MIN_TRUNCATED_CHARS:
        return False
    return all(token in _GENERIC_SUFFIX_MARKERS for token in name_tokens[len(company_tokens):])


@dataclass(frozen=True)
class ScopeVerdict:
    """Where a person's current employer sits against the allowlist.

    ``in_scope`` is True (a listed employer, named in ``employer``), False (an
    identified current employer that is not listed), or None (no current
    employer identified: the judge decides). ``field`` and ``value`` name the
    evidence.
    """

    in_scope: bool | None
    employer: str = ""
    field: str = ""
    value: str = ""


@dataclass(frozen=True)
class EmployerAllowlist:
    facet_groups: tuple[tuple[str, ...], ...]
    aliases: Mapping[str, str] = field(default_factory=dict)
    # Listed employers that count only in a title-family seat, and the block
    # that defines the families. Empty for a brief without ``title_restricted``.
    restricted: frozenset[str] = frozenset()
    facet_titles: tuple[str, ...] = ()
    title_markers: tuple[str, ...] = ()
    title_exclusions: tuple[str, ...] = ()

    def is_restricted(self, name: object) -> bool:
        """Whether a listed employer counts only in a title-family seat."""

        return bool(name) and str(name) in self.restricted

    def title_in_family(self, title: object) -> bool:
        """Whether a job title or headline is in the restricted employers' families."""

        tokens = _fold_tokens(title)
        if not tokens or not self.title_markers:
            return False
        named = [
            marker
            for marker in self.title_markers
            if _contains_window(tokens, _fold_tokens(marker))
        ]
        if not named:
            return False
        excluded = [
            word
            for word in self.title_exclusions
            if _contains_window(tokens, _fold_tokens(word))
        ]
        return all(
            any(_contains_window(_fold_tokens(marker), _fold_tokens(word)) for marker in named)
            for word in excluded
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Every listed employer, in group order."""

        seen: list[str] = []
        for group in self.facet_groups:
            for name in group:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    def _spellings(self) -> list[tuple[str, str]]:
        """(spelling, listed employer) for every name and alias."""

        pairs = [(name, name) for name in self.names]
        pairs.extend((alias, canonical) for alias, canonical in self.aliases.items())
        return pairs

    def canonical_for(self, company: object) -> str | None:
        """The listed employer a company or facet value names, else None."""

        company_tokens = _fold_tokens(_strip_employment_suffix(company))
        if not company_tokens:
            return None
        for spelling, canonical in self._spellings():
            name_tokens = _fold_tokens(spelling)
            if _contains_window(company_tokens, name_tokens):
                return canonical
        for spelling, canonical in self._spellings():
            if _is_truncation(company_tokens, _fold_tokens(spelling)):
                return canonical
        return None

    def listed_entity(self, value: object) -> str | None:
        """The listed employer a LinkedIn facet entity IS, else None.

        Stricter than :meth:`canonical_for`, because it judges what landed on
        the rail: the typeahead can take a near miss onto a different company
        ("Handshake" landing as "Handshake Partners"). The entity counts when
        its name equals a listed name or alias once trailing corporate
        suffixes ("AI", "Inc", "LLC", "Technologies") are set aside on both
        sides, so "Cogito Tech LLC" is Cogito Tech and "Uber" is not Uber AI
        Solutions.
        """

        core = _core_tokens(_fold_tokens(_strip_employment_suffix(value)))
        if not core:
            return None
        for spelling, canonical in self._spellings():
            spelled = _core_tokens(_fold_tokens(spelling))
            if core == spelled or "".join(core) == "".join(spelled):
                return canonical
        return None

    def mentioned_in(self, text: object) -> set[str]:
        """Every listed employer a free text names, by name or alias."""

        tokens = _fold_tokens(text)
        return {
            canonical
            for spelling, canonical in self._spellings()
            if _contains_window(tokens, _fold_tokens(spelling))
        }

    def headline_employer(self, headline: object) -> str | None:
        """The listed employer a headline names as CURRENT ("at X", "@ X")."""

        text = str(headline or "")
        if not text:
            return None
        for spelling, canonical in self._spellings():
            if match_current_employer("", text, [spelling]) is not None:
                return canonical
        return None


def is_placeholder_company(company: object) -> bool:
    """Whether a company field names no employer ("Self-employed", "Stealth")."""

    tokens = _fold_tokens(_strip_employment_suffix(company))
    return not tokens or " ".join(tokens) in _PLACEHOLDER_COMPANY_MARKERS


def current_employer_scope(
    *,
    companies: Iterable[object],
    headlines: Iterable[object],
    allowlist: EmployerAllowlist,
    company_field: str = "current_company",
    titles: Iterable[object] = (),
) -> ScopeVerdict:
    """Whether a person's CURRENT employers put them inside the allowlist.

    ``companies`` are the current employer fields (a card's current company,
    the company of every position marked Present); ``headlines`` are matched
    only where they name an employer as current; ``titles`` are the current
    job titles. Any listed, unrestricted employer anywhere among them is in
    scope. A restricted employer is in scope only when a current title or a
    headline is in its title families; with titles on record and none in the
    families it is out of scope (field ``current_title``), and with no title
    on record the judge decides. Otherwise the first identified employer is
    out of scope; with none identified, the verdict is None.
    """

    company_list = [str(company or "").strip() for company in companies]
    headline_list = [str(headline or "") for headline in headlines]
    restricted_hit: tuple[str, str, str] | None = None
    for company in company_list:
        if not company or is_placeholder_company(company):
            continue
        canonical = allowlist.canonical_for(company)
        if canonical and not allowlist.is_restricted(canonical):
            return ScopeVerdict(True, canonical, company_field, company)
        if canonical and restricted_hit is None:
            restricted_hit = (canonical, company_field, company)
    for headline in headline_list:
        canonical = allowlist.headline_employer(headline)
        if canonical and not allowlist.is_restricted(canonical):
            return ScopeVerdict(True, canonical, "headline", headline)
        if canonical and restricted_hit is None:
            restricted_hit = (canonical, "headline", headline)
    if restricted_hit is not None:
        canonical, hit_field, hit_value = restricted_hit
        seats = [str(title or "").strip() for title in titles]
        seats = [seat for seat in seats if seat]
        if any(allowlist.title_in_family(text) for text in [*seats, *headline_list]):
            return ScopeVerdict(True, canonical, hit_field, hit_value)
        if seats:
            employer = _strip_employment_suffix(hit_value) if hit_field != "headline" else canonical
            return ScopeVerdict(False, canonical, "current_title", f"{seats[0]} at {employer}")
        return ScopeVerdict(None, canonical, hit_field, hit_value)
    for company in company_list:
        if company and not is_placeholder_company(company):
            return ScopeVerdict(False, "", company_field, company)
    return ScopeVerdict(None)


def present_entry_company(entry: object) -> str | None:
    """The company of a card's "Title at Company (YYYY – Present)" line, else None."""

    text = str(entry or "").strip()
    match = _PRESENT_TAIL.search(text)
    if not match:
        return None
    head = text[: match.start()].strip()
    if " at " not in head:
        return None
    company = head.rsplit(" at ", 1)[1].strip()
    return company or None


def present_entry_title(entry: object) -> str | None:
    """The title of a card's "Title at Company (YYYY – Present)" line, else None."""

    text = str(entry or "").strip()
    match = _PRESENT_TAIL.search(text)
    if not match:
        return None
    head = text[: match.start()].strip()
    if " at " not in head:
        return None
    title = head.rsplit(" at ", 1)[0].strip()
    return title or None


def _string_list(value: Any, *, allow_empty: bool) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) and item.strip() for item in value):
        return None
    if not value and not allow_empty:
        return None
    return [item.strip() for item in value]


def _parse_title_restricted(
    value: Any, groups: list[tuple[str, ...]]
) -> dict[str, Any] | None:
    """The block's fields, or None when it is malformed."""

    if not isinstance(value, dict):
        return None
    if any(key not in _TITLE_RESTRICTED_KEYS for key in value if not str(key).startswith("_")):
        return None
    employers = _string_list(value.get("employers"), allow_empty=False)
    facet_titles = _string_list(value.get("facet_titles"), allow_empty=False)
    markers = _string_list(value.get("title_markers"), allow_empty=False)
    exclusions = _string_list(value.get("title_exclusions", []), allow_empty=True)
    if employers is None or facet_titles is None or markers is None or exclusions is None:
        return None
    if len(facet_titles) > FACET_COMPANY_VALUE_CAP:
        return None
    names = {name for group in groups for name in group}
    restricted = frozenset(employers)
    if not restricted <= names:
        return None
    # A group carrying a restricted employer carries nothing else, so a facet
    # never bounds competitors by the restricted employers' titles.
    for group in groups:
        members = set(group)
        if members & restricted and not members <= restricted:
            return None
    fields = {
        "restricted": restricted,
        "facet_titles": tuple(facet_titles),
        "title_markers": tuple(markers),
        "title_exclusions": tuple(exclusions),
    }
    probe = EmployerAllowlist(facet_groups=tuple(groups), **fields)
    if not all(probe.title_in_family(title) for title in facet_titles):
        return None
    return fields


def parse_employer_allowlist(value: Any) -> EmployerAllowlist | None:
    """The field as an :class:`EmployerAllowlist`, or None when absent or malformed."""

    if not isinstance(value, dict):
        return None
    raw_groups = value.get("facet_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        return None
    groups: list[tuple[str, ...]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not raw_group:
            return None
        if len(raw_group) > FACET_COMPANY_VALUE_CAP:
            return None
        if not all(isinstance(name, str) and name.strip() for name in raw_group):
            return None
        groups.append(tuple(name.strip() for name in raw_group))
    names = {name for group in groups for name in group}
    raw_aliases = value.get("aliases", {})
    if not isinstance(raw_aliases, dict):
        return None
    aliases: dict[str, str] = {}
    for alias, canonical in raw_aliases.items():
        if not isinstance(alias, str) or not isinstance(canonical, str):
            return None
        if alias.startswith("_"):
            continue
        if canonical.strip() not in names or not alias.strip():
            return None
        aliases[alias.strip()] = canonical.strip()
    restricted: dict[str, Any] = {}
    if "title_restricted" in value:
        parsed = _parse_title_restricted(value["title_restricted"], groups)
        if parsed is None:
            return None
        restricted = parsed
    return EmployerAllowlist(facet_groups=tuple(groups), aliases=aliases, **restricted)


def employer_allowlist_is_valid(value: Any) -> bool:
    return parse_employer_allowlist(value) is not None


def employer_allowlist_from_brief(brief: Any) -> EmployerAllowlist | None:
    """The brief's allowlist, read from ``brief.raw`` on the seed and the generated brief."""

    raw = getattr(brief, "raw", None)
    if not isinstance(raw, dict):
        return None
    return parse_employer_allowlist(raw.get(EMPLOYER_ALLOWLIST_KEY))


__all__ = [
    "EMPLOYER_ALLOWLIST_KEY",
    "EMPLOYER_SCOPE_GATE_PATH",
    "FACET_COMPANY_VALUE_CAP",
    "EmployerAllowlist",
    "ScopeVerdict",
    "current_employer_scope",
    "employer_allowlist_from_brief",
    "employer_allowlist_is_valid",
    "is_placeholder_company",
    "parse_employer_allowlist",
    "present_entry_company",
    "present_entry_title",
]
