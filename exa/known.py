"""Cross-campaign "already known" lookup — report-only, never a filter.

Two independent instruments converging on the same people is
evidence the calibration is encoded consistently, and a recruiter
reading the slate needs to know which names are already in a pipeline
before spending a reach-out on them.

Three properties make this safe to be wrong about:

1. **It never drops anybody.** A match moves a person into the report's
   "already known" bucket and nothing else. Matching is name-based, so a
   false positive is inevitable eventually; the cost of one is a line in
   the wrong section, not a lost candidate.
2. **It is best effort.** A sibling state dir that is missing, locked,
   mid-migration, or written by a schema this version doesn't know
   yields nothing and logs nothing to the console.
3. **It opens other campaigns' databases READ-ONLY.** Constructing a
   :class:`~shared.runtime_state.store.RuntimeStateStore` would run
   ``initialize()`` → ``_migrate()`` — a WRITE against a live campaign's
   canonical state — which is not a thing a report lookup gets to do.
   The handle here is ``mode=ro`` and the query is a single SELECT.

Only SAVE-family verdicts count as "known". A person another campaign
REJECTED is not known to the recruiter as a candidate; surfacing them as
one would quietly import that campaign's rejection into this one, and
the briefs are different.

Matching is by public LinkedIn handle when both sides carry one, and by
name when either side lacks one. A LinkedIn Recruiter save carries a
``/talent/`` URL with no public handle, and Recruiter shows some people
as "Larissa M.". An initial stands for a full surname only when the first
names are equal, the initial fits, and exactly one person on the other
side fits it; an ambiguous initial matches nobody and is logged.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from exa.identity import _PUNCT, _fold_ascii, _name_keys
from exa.schemas import ExaCandidate
from shared.identity_resolution import (
    normalize_person_name,
    normalize_public_linkedin_handle,
)
from shared.save_destination import SAVE_DECISION_FAMILY


logger = logging.getLogger(__name__)

# The candidates a sibling campaign considers saved. Read from the
# canonical column, not from a projection file, because a projection can
# lag and the whole point of the bucket is "already in the pipeline".
_KNOWN_QUERY = """
    SELECT display_name, profile_url, identity_key, terminal_decision
    FROM candidates
    WHERE terminal_decision IN ({placeholders})
"""


@dataclass
class KnownPerson:
    """One person another campaign already holds."""

    display_name: str
    profile_url: str
    label: str
    decision: str = ""


def _first_and_surname(name: str) -> tuple[str, str] | None:
    """``(first, surname)`` as normalized tokens, or None.

    The same normalization as :func:`exa.identity._name_keys`: diacritics
    folded, lowercased, punctuation and credentials stripped. A surname
    of one letter is an initial ("Larissa M." -> ``("larissa", "m")``).
    A first name of one letter is not a first name to match on.
    """

    tokens = _PUNCT.sub(" ", normalize_person_name(_fold_ascii(name))).split()
    if len(tokens) < 2 or len(tokens[0]) < 2:
        return None
    return tokens[0], tokens[-1]


@dataclass
class KnownPeople:
    """Name- and handle-indexed view over every sibling campaign read."""

    by_handle: dict[str, KnownPerson] = field(default_factory=dict)
    by_name_key: dict[str, KnownPerson] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    # The name index over the people with NO public handle: the only ones a
    # candidate that has a handle may be matched to by name.
    by_name_key_without_handle: dict[str, KnownPerson] = field(default_factory=dict)
    # One line per initial that fit more than one person, for the log.
    ambiguous: list[str] = field(default_factory=list)
    # (first name, surname initial) -> [(surname, person, has_handle)].
    _by_initial: dict[tuple[str, str], list[tuple[str, KnownPerson, bool]]] = field(
        default_factory=dict, repr=False
    )
    # This campaign's own people, (first name, surname initial) -> surnames,
    # once :meth:`set_pool` is called: the other side for a known initial.
    _pool: dict[tuple[str, str], set[str]] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len({id(person) for person in self.by_handle.values()} |
                   {id(person) for person in self.by_name_key.values()})

    def __bool__(self) -> bool:
        return bool(self.by_handle or self.by_name_key)

    def add(self, person: KnownPerson) -> None:
        handle = normalize_public_linkedin_handle(person.profile_url or "")
        if handle:
            self.by_handle.setdefault(handle, person)
        for key in _name_keys(person.display_name or ""):
            self.by_name_key.setdefault(key, person)
            if not handle:
                self.by_name_key_without_handle.setdefault(key, person)
        parts = _first_and_surname(person.display_name or "")
        if parts is not None:
            first, surname = parts
            self._by_initial.setdefault((first, surname[0]), []).append(
                (surname, person, bool(handle))
            )

    def set_pool(self, names: Iterable[str]) -> None:
        """Name everybody this campaign holds, so a known initial can be checked.

        A known "Larissa M." fits a found "Larissa Moreau" only when
        nobody else in this campaign is a Larissa M-something. Without a
        pool, only the known side is checked.
        """

        pool: dict[tuple[str, str], set[str]] = {}
        for name in names:
            parts = _first_and_surname(name or "")
            if parts is None or len(parts[1]) < 2:
                continue
            first, surname = parts
            pool.setdefault((first, surname[0]), set()).add(surname)
        self._pool = pool

    def match(self, candidate: ExaCandidate) -> str | None:
        """The label of the campaign that already holds this person.

        A candidate that carries a public LinkedIn handle is matched on the
        handle first. A handle that is absent from the index is positive
        evidence against every known person who HAS a handle: the same name
        with a different handle is a different person, and filing a net-new
        candidate under "already known, no action" is the failure to avoid.
        It says nothing about a known person with no handle (a Recruiter
        save), so the name is compared against those.

        A candidate with no handle has no identity to compare, so the name
        is compared against everybody. Names go through the same
        ``_name_keys`` normalizer the do-not-save gate uses ("Avery L.
        Holt" and "Holt, Avery" reach the same key as "Avery Holt"), then
        through the initial rule in the module docstring.
        """

        person = self.match_person(candidate)
        return person.label if person is not None else None

    def match_person(self, candidate: ExaCandidate) -> KnownPerson | None:
        handle = normalize_public_linkedin_handle(
            candidate.linkedin_url or ""
        ) or normalize_public_linkedin_handle(candidate.identity_key or "")
        if handle:
            person = self.by_handle.get(handle)
            if person is not None:
                return person
            names = self.by_name_key_without_handle
        else:
            names = self.by_name_key
        for key in _name_keys(candidate.name or ""):
            if key in names:
                return names[key]
        return self._match_initial(candidate.name or "", with_handles=not handle)

    def _match_initial(self, name: str, *, with_handles: bool) -> KnownPerson | None:
        parts = _first_and_surname(name)
        if parts is None:
            return None
        first, surname = parts
        fits = [
            (known_surname, person)
            for known_surname, person, has_handle in self._by_initial.get(
                (first, surname[0]), []
            )
            if with_handles or not has_handle
        ]
        if len(surname) == 1:
            # The candidate is the initial: one known full name must fit it.
            full = sorted({known for known, _ in fits if len(known) > 1})
            if len(full) > 1:
                self._note_ambiguous(name, len(full), "known")
            if len(full) != 1:
                return None
            return next(person for known, person in fits if known == full[0])
        initials = [person for known, person in fits if len(known) == 1]
        if not initials:
            return None
        # The known person is the initial: this candidate must be the only
        # one it could stand for, in this campaign and among the known.
        rivals = {known for known, _ in fits if len(known) > 1 and known != surname}
        if self._pool is not None:
            rivals |= self._pool.get((first, surname[0]), set()) - {surname}
        if rivals:
            self._note_ambiguous(
                initials[0].display_name, len(rivals) + 1, "found or known"
            )
            return None
        return initials[0]

    def _note_ambiguous(self, name: str, count: int, where: str) -> None:
        line = (
            f"already-known check: {name!r} fits {count} {where} people; "
            "matched to none"
        )
        if line not in self.ambiguous:
            self.ambiguous.append(line)
            logger.info(line)


def load_known_people(state_dirs: Sequence[Path] | Iterable[Path]) -> KnownPeople:
    """Read every sibling state dir's saved candidates, fail-soft.

    The label a match reports is the state dir's own directory name —
    the thing a recruiter recognizes ("9000000006", "ciso") without
    needing the full path.
    """

    known = KnownPeople()
    for raw_dir in state_dirs or []:
        state_dir = Path(raw_dir)
        label = state_dir.name or str(state_dir)
        rows = _read_saved_rows(state_dir / "runtime_state.sqlite3")
        if not rows:
            continue
        known.labels.append(label)
        for row in rows:
            known.add(
                KnownPerson(
                    display_name=str(row["display_name"] or ""),
                    profile_url=str(row["profile_url"] or row["identity_key"] or ""),
                    label=label,
                    decision=str(row["terminal_decision"] or ""),
                )
            )
    return known


def _read_saved_rows(db_path: Path) -> list[sqlite3.Row]:
    if not db_path.exists():
        return []
    decisions = sorted(SAVE_DECISION_FAMILY)
    sql = _KNOWN_QUERY.format(placeholders=",".join("?" for _ in decisions))
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return list(conn.execute(sql, decisions).fetchall())
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.debug("known-people read failed for %s: %s", db_path, exc)
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass


__all__ = [
    "KnownPeople",
    "KnownPerson",
    "load_known_people",
]
