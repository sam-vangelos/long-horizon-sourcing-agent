"""Current-employer scope for Exa briefs that restrict saves to named firms.

A competitors-only brief sources people who CURRENTLY work at one of its target
firms. The judge carries that rule as prompt text; this module is the
deterministic check behind it. It runs AFTER judgment and never drops anyone:
a save whose current employer is positively identified and is not in scope
moves to the review bucket with the reason "scope: current employer X". A
record that names no employer keeps its verdict, because a sparse page is not
evidence that the person left. The stored verdict is the judge's and is never
rewritten; the slate re-applies the check each time it renders.

The check is inert unless the brief opts in with the top-level block
``exa_current_employer_scope``::

    {"aliases": {"Outlier": "Scale AI", "DataAnnotation": "Surge AI"}}

In scope is every ``exa_target_firms`` name and alias, plus the block's own
aliases. Those map a contractor brand or a business-unit name to its parent
firm for scope only: they must not be a target firm's name or alias, and the
lint refuses one that appears in any compiled query, so a contractor brand
never becomes a retrieval term.

Matching is :func:`shared.employer_blacklist.match_current_employer` over the
signals :func:`exa.employer_gate.current_employer_signals` reads, the same
current-employer rule the blacklist gate applies: a company field matches on a
word boundary, a headline only where it says "at X" or "@ X". Any in-scope
signal puts the person in scope; a contract seat at a competitor counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from shared.employer_blacklist import match_current_employer

from exa.employer_gate import current_employer_signals
from exa.schemas import ExaCandidate


SCOPE_FIELD = "exa_current_employer_scope"
ALIASES_KEY = "aliases"
SCOPE_KEYS = (ALIASES_KEY,)

IN_SCOPE = "in_scope"
OUT_OF_SCOPE = "out_of_scope"
UNIDENTIFIED = "unidentified"


def _flatten(value: object) -> str:
    return " ".join(str(value or "").split())


def _scope_block(brief: Any) -> Any:
    """The block off a raw dict or either loaded brief shape; None when absent."""

    if isinstance(brief, Mapping):
        return brief.get(SCOPE_FIELD)
    raw = getattr(brief, "raw", None)
    if isinstance(raw, Mapping):
        return raw.get(SCOPE_FIELD)
    return getattr(brief, SCOPE_FIELD, None)


def _target_firms(brief: Any) -> list[dict]:
    firms = (
        brief.get("exa_target_firms")
        if isinstance(brief, Mapping)
        else getattr(brief, "exa_target_firms", None)
    )
    if not isinstance(firms, list):
        return []
    return [
        firm
        for firm in firms
        if isinstance(firm, dict) and isinstance(firm.get("name"), str) and firm["name"].strip()
    ]


def _firm_labels(firm: dict) -> list[str]:
    labels = [_flatten(firm["name"])]
    for alias in firm.get("aliases") or []:
        if isinstance(alias, str) and alias.strip() and _flatten(alias) not in labels:
            labels.append(_flatten(alias))
    return labels


def scope_errors(brief: Any, *, query_texts: Iterable[str] = ()) -> list[str]:
    """What is wrong with the brief's scope block, as short phrases.

    Empty when the block is absent or well formed. ``query_texts`` are the
    compiled queries' text; a scope alias found in one is an error, because a
    contractor brand used as a retrieval term pulls its contractor pool.
    """

    block = _scope_block(brief)
    if block is None:
        return []
    if not isinstance(block, dict):
        return [f"{SCOPE_FIELD} must be an object"]
    errors = [f"unknown key {key!r}" for key in block if key not in SCOPE_KEYS]

    firms = _target_firms(brief)
    if not firms:
        errors.append("the scope has no target firms to hold anyone in it")
    names = {_flatten(firm["name"]).casefold(): _flatten(firm["name"]) for firm in firms}
    query_labels = {label.casefold() for firm in firms for label in _firm_labels(firm)}

    aliases = block.get(ALIASES_KEY, {})
    if not isinstance(aliases, dict):
        errors.append("aliases must be an object mapping an alias to its target firm")
        return errors
    texts = [str(text or "") for text in query_texts]
    for alias, parent in aliases.items():
        if not (
            isinstance(alias, str) and alias.strip() and isinstance(parent, str) and parent.strip()
        ):
            errors.append("aliases must map non-empty names to non-empty target firm names")
            continue
        if firms and _flatten(parent).casefold() not in names:
            errors.append(f"alias {alias!r} maps to {parent!r}, which is not a target firm")
        if _flatten(alias).casefold() in query_labels:
            errors.append(
                f"alias {alias!r} is already a query term (a target firm's name or alias); "
                "scope aliases never become retrieval terms"
            )
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(_flatten(alias))}(?![A-Za-z0-9])", re.I)
        for text in texts:
            if pattern.search(text):
                errors.append(
                    f"alias {alias!r} appears in a compiled query; scope aliases "
                    "never become retrieval terms"
                )
                break
    return errors


@dataclass(frozen=True)
class ScopeVerdict:
    """Where the record says the person works now, against the scope."""

    status: str
    employer: str = ""
    parent: str = ""

    @property
    def note(self) -> str:
        if self.status != OUT_OF_SCOPE:
            return ""
        return f"scope: current employer {self.employer}"


@dataclass(frozen=True)
class CurrentEmployerScope:
    """Every name that counts as in scope, each mapped to its target firm."""

    parents: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_brief(cls, brief: Any) -> CurrentEmployerScope | None:
        """The scope, or None when the brief does not opt in.

        Raises ``ValueError`` on a malformed block rather than running
        without it: the lint refuses such a brief before any spend.
        """

        if _scope_block(brief) is None:
            return None
        errors = scope_errors(brief)
        if errors:
            raise ValueError(f"{SCOPE_FIELD} in the brief is malformed: " + "; ".join(errors))
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(label: str, parent: str) -> None:
            key = label.casefold()
            if key not in seen:
                seen.add(key)
                pairs.append((label, parent))

        firms = _target_firms(brief)
        for firm in firms:
            for label in _firm_labels(firm):
                add(label, _flatten(firm["name"]))
        canonical = {_flatten(firm["name"]).casefold(): _flatten(firm["name"]) for firm in firms}
        for alias, parent in (_scope_block(brief).get(ALIASES_KEY) or {}).items():
            add(_flatten(alias), canonical[_flatten(parent).casefold()])
        return cls(parents=tuple(pairs))

    @property
    def names(self) -> list[str]:
        return [label for label, _parent in self.parents]

    def _parent_of(self, label: str) -> str:
        key = label.casefold()
        return next((parent for name, parent in self.parents if name.casefold() == key), label)

    def check(self, candidate: ExaCandidate) -> ScopeVerdict:
        """In scope on any in-scope signal; out only on an identified other employer."""

        companies, headlines = current_employer_signals(candidate)
        names = self.names
        for company in companies:
            hit = match_current_employer(company, "", names)
            if hit is not None:
                return ScopeVerdict(IN_SCOPE, employer=company, parent=self._parent_of(hit[0]))
        for headline in headlines:
            hit = match_current_employer("", headline, names)
            if hit is not None:
                return ScopeVerdict(IN_SCOPE, employer=headline, parent=self._parent_of(hit[0]))
        if companies:
            return ScopeVerdict(OUT_OF_SCOPE, employer=companies[0])
        return ScopeVerdict(UNIDENTIFIED)


__all__ = [
    "ALIASES_KEY",
    "IN_SCOPE",
    "OUT_OF_SCOPE",
    "SCOPE_FIELD",
    "UNIDENTIFIED",
    "CurrentEmployerScope",
    "ScopeVerdict",
    "scope_errors",
]
