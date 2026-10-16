"""Deterministic lexicon-driven enrichment of one Boolean string.

The scorer says how good a string is; this says what to do about the part of
the gap that is mechanical. It adds the surface forms a concept already in a
group is missing, adds the morphology and hyphen twins of terms already
there, and drops forms the lexicon marks ``rare``. It invents nothing: every
term it writes was already declared for a concept the author chose.

WHAT IT WILL NOT DO, and why each one is a rule rather than a preference:

- It never touches a NOT group. Widening an exclusion excludes more people.
- It never removes an authored term that is not ``rare``, and never removes
  even a rare one when it is the only surface in its group naming its
  concept — dropping the last representative changes what the group means.
- It never changes group structure or operators. It rewrites the interior of
  one OR group at a time, between the first and last term the parser found
  there, and only when that span holds nothing but those terms and the ``OR``
  between them.
- It never adds an umbrella term, a noisy term, a generic verb, an employer
  name or a ``rare`` form. The first four are charged by the scorer as wasted
  gates or swallowed groups; the fifth is what this function removes, and
  adding one would make the pass non-idempotent.
- It never adds a surface the scorer's own structure checks would charge for:
  a phrase carrying a bare article, a stem-looking tail, a dropped special
  character, a comma numeral. A derived hyphen twin can produce exactly such
  a surface — "LLM-as-a-judge" unhyphenates into a sentence fragment — so
  every candidate is screened before it is written.
- It never exceeds ``max_chars``, which defaults to the scorer's own
  2000-character length cap, because passing that cap is itself a charge.

Idempotence is a postcondition, not an aspiration: ``enrich(enrich(x))``
equals ``enrich(x)``. Candidates are generated to a fixed point within one
call, so a form added on this pass has already contributed its own twins.

Safety is checked twice. Each group must pass preconditions before it is
rewritten, and the finished string is re-parsed and compared group by group
against the original; any structural difference discards the whole result and
returns the input untouched. An illegal input is returned unchanged with
``legal=False``.

Standard library only, plus this package's own parser and lexicon; nothing
here imports from the rest of the repo, so the package stays vendorable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .lexicon import RARE_REGISTER, Lexicon, normalize
from .parser import And, Node, Not, Or, ParseResult, Term, parse

# The scorer's own length cap: past it the tail of the keyword field may be
# dropped, and the scorer charges for it. Enrichment stops rather than earn
# that charge.
DEFAULT_MAX_CHARS = 2000

# Mirrors score.py's structure checks. A candidate that would trip any of
# them is not worth its coverage credit.
_ARTICLE_KEYS = frozenset({"a", "an", "the", "my", "our", "their", "your"})
_MID_WORD_STEM_SUFFIX_KEYS = ("nc", "iz", "yz")
_MID_WORD_STEM_EXCEPTION_KEYS = frozenset(
    {"zinc", "quiz", "showbiz", "biz", "inc", "sync", "async", "func"}
)
_NOOP_SPECIAL_CHARACTERS = frozenset("$%&+")
_COMMA_NUMERAL_RE = re.compile(r"\d,\d")
# What may legally sit between two terms of a plain OR group.
_OR_JOIN_RE = re.compile(r"^\s+OR\s+$")


@dataclass(frozen=True)
class GroupChange:
    """What happened inside one OR group."""

    group_index: int
    forms_added: tuple[str, ...] = ()
    twins_added: tuple[str, ...] = ()
    rare_removed: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.forms_added or self.twins_added or self.rare_removed)

    def to_dict(self) -> dict:
        return {
            "group_index": self.group_index,
            "forms_added": list(self.forms_added),
            "twins_added": list(self.twins_added),
            "rare_removed": list(self.rare_removed),
        }


@dataclass(frozen=True)
class EnrichReport:
    original: str
    enriched: str
    legal: bool
    changes: tuple[GroupChange, ...] = ()
    truncated: bool = False
    rejected: str = ""

    @property
    def unchanged(self) -> bool:
        return self.enriched == self.original

    def to_dict(self) -> dict:
        return {
            "enriched": self.enriched,
            "legal": self.legal,
            "unchanged": self.unchanged,
            "truncated": self.truncated,
            "changes": [change.to_dict() for change in self.changes],
            **({"rejected": self.rejected} if self.rejected else {}),
        }


# ---------------------------------------------------------------------------
# Group discovery
# ---------------------------------------------------------------------------


@dataclass
class _Group:
    terms: tuple[Term, ...]
    negated: bool
    index: int
    enrichable: bool = False
    kept: list[Term] = field(default_factory=list)
    additions: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    added_forms: list[str] = field(default_factory=list)
    added_twins: list[str] = field(default_factory=list)


def _collect(node: Node, negated: bool, out: list[tuple[tuple[Term, ...], bool]]) -> None:
    """The scorer's group decomposition, restated so enrichment does not
    depend on a private helper of the module that scores it."""
    if isinstance(node, Term):
        out.append(((node,), negated))
        return
    if isinstance(node, Or):
        terms = tuple(child for child in node.children if isinstance(child, Term))
        if terms:
            out.append((terms, negated))
        for child in node.children:
            if not isinstance(child, Term):
                _collect(child, negated, out)
        return
    if isinstance(node, And):
        for child in node.children:
            _collect(child, negated, out)
        return
    if isinstance(node, Not):
        _collect(node.child, True, out)


def _has_and(node: Node | None) -> bool:
    if isinstance(node, And):
        return True
    if isinstance(node, Or):
        return any(_has_and(child) for child in node.children)
    if isinstance(node, Not):
        return _has_and(node.child)
    return False


def _terms_are_contiguous(source: str, terms: Sequence[Term]) -> bool:
    """True when only whitespace and ``OR`` sit between consecutive terms.

    An ``Or`` node can hold a nested group between two of its direct terms.
    Rewriting the span from the first term to the last would then clobber
    that nested group, so such a group is left alone.
    """
    for earlier, later in zip(terms, terms[1:]):
        if earlier.span[1] < 0 or later.span[0] < 0:
            return False
        if not _OR_JOIN_RE.match(source[earlier.span[1] : later.span[0]]):
            return False
    return True


def _closes_safely(source: str, terms: Sequence[Term], root_is_group: bool) -> bool:
    """True when appending ``OR x`` after the last term cannot re-associate.

    Inside parentheses it never can. At the top level it can only when the
    whole expression is that one OR group, with no AND anywhere to bind
    tighter than the OR being extended.
    """
    end = terms[-1].span[1]
    tail = source[end:].lstrip()
    if tail.startswith(")"):
        return True
    return root_is_group and not tail


# ---------------------------------------------------------------------------
# Candidate screening
# ---------------------------------------------------------------------------


def _would_be_charged(surface: str) -> bool:
    """True when the scorer's structure checks would charge for this term."""
    tokens = surface.split()
    if not tokens:
        return True
    if any(token.lower() in _ARTICLE_KEYS for token in tokens):
        return True
    last = tokens[-1].lower()
    if last.endswith("-"):
        return True
    if last not in _MID_WORD_STEM_EXCEPTION_KEYS and any(
        last.endswith(suffix) for suffix in _MID_WORD_STEM_SUFFIX_KEYS
    ):
        return True
    if any(char in _NOOP_SPECIAL_CHARACTERS for char in surface):
        return True
    if _COMMA_NUMERAL_RE.search(surface):
        return True
    return '"' in surface


def _may_add(
    surface: str,
    *,
    lexicon: Lexicon,
    present: set[str],
    group_concepts: set[str],
) -> bool:
    key = normalize(surface)
    if not key or key in present:
        return False
    if _would_be_charged(key):
        return False
    if lexicon.is_umbrella(key) or lexicon.is_noisy(key):
        return False
    if lexicon.is_generic_verb(key) or lexicon.is_employer(key):
        return False
    form = lexicon.form_for(key)
    if form is not None and form.register == RARE_REGISTER:
        # Adding what this pass removes would never settle.
        return False
    concept = lexicon.concept_for(key)
    if concept is not None and concept.id not in group_concepts:
        # A twin owned by a different concept would widen the group's meaning
        # and can charge the repeated-concept bloat across an AND.
        return False
    return True


def _candidates_for(
    group: _Group, lexicon: Lexicon, present: set[str]
) -> tuple[list[str], list[str]]:
    """Missing common forms and missing twins, to a fixed point.

    A form added here may itself declare twins, so the worklist runs until it
    stops growing; that is what makes a second pass a no-op.
    """
    group_concepts = {
        concept.id
        for term in group.terms
        if (concept := lexicon.concept_for(term.text)) is not None
    }
    forms: list[str] = []
    twins: list[str] = []
    seen = set(present)

    pending = [term.text for term in group.terms]
    while pending:
        surface = pending.pop(0)
        concept = lexicon.concept_for(surface)
        if concept is None:
            continue
        for form in concept.common_forms:
            key = form.normalized
            if key in seen:
                continue
            if not _may_add(
                key, lexicon=lexicon, present=seen, group_concepts=group_concepts
            ):
                continue
            forms.append(key)
            seen.add(key)
            pending.append(key)
        if concept.proper_noun:
            # The doctrine forbids fabricated variants for named tools.
            continue
        for twin in sorted(lexicon.twins_for(surface)):
            if twin in seen:
                continue
            if not _may_add(
                twin, lexicon=lexicon, present=seen, group_concepts=group_concepts
            ):
                continue
            twins.append(twin)
            seen.add(twin)
            pending.append(twin)
    return forms, twins


def _removable_rare(
    group: _Group, lexicon: Lexicon, additions: Sequence[str]
) -> list[Term]:
    """Rare forms safe to drop: never the last surface naming their concept.

    The additions count as siblings. They are the whole point — replacing a
    rare spelling with the common ones is the trade this function exists to
    make — and counting them is also what makes the pass idempotent: judging
    removability against the authored terms alone leaves a rare form standing
    on the first pass and removable on the second.
    """
    by_concept: dict[str, list[Term]] = {}
    for term in group.terms:
        concept = lexicon.concept_for(term.text)
        if concept is not None:
            by_concept.setdefault(concept.id, []).append(term)
    added_by_concept: dict[str, int] = {}
    for surface in additions:
        concept = lexicon.concept_for(surface)
        if concept is not None:
            added_by_concept[concept.id] = added_by_concept.get(concept.id, 0) + 1

    doomed: list[Term] = []
    for term in group.terms:
        form = lexicon.form_for(term.text)
        if form is None or form.register != RARE_REGISTER:
            continue
        concept = lexicon.concept_for(term.text)
        if concept is None:
            continue
        siblings = [
            other
            for other in by_concept.get(concept.id, [])
            if other is not term and other not in doomed
        ]
        if not siblings and not added_by_concept.get(concept.id):
            continue
        doomed.append(term)
    if len(doomed) >= len(group.terms) + len(additions):
        return []
    return doomed


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_authored(source: str, term: Term) -> str:
    """The term exactly as the author wrote it, quoting and all."""
    start, end = term.span
    if start < 0 or end < 0:
        return f'"{term.text}"' if term.quoted else term.text
    return source[start:end]


def _render_addition(surface: str, quoted: bool) -> str:
    if quoted or " " in surface:
        # An unquoted multiword run tokenizes as separate terms and the
        # scorer charges for it, so a phrase is always quoted.
        return f'"{surface}"'
    return surface


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _signature(result: ParseResult, lexicon: Lexicon) -> list[tuple[bool, frozenset[str]]]:
    raw: list[tuple[tuple[Term, ...], bool]] = []
    if result.tree is not None:
        _collect(result.tree, False, raw)
    return [
        (negated, frozenset(term.normalized for term in terms)) for terms, negated in raw
    ]


def _structure_preserved(
    before: ParseResult, after: ParseResult, lexicon: Lexicon, dropped: set[str]
) -> bool:
    if not after.legal:
        return False
    old = _signature(before, lexicon)
    new = _signature(after, lexicon)
    if len(old) != len(new):
        return False
    for (old_neg, old_terms), (new_neg, new_terms) in zip(old, new):
        if old_neg != new_neg:
            return False
        if not (old_terms - dropped) <= new_terms:
            return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def enrich(
    boolean: str,
    lexicon: Lexicon,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> EnrichReport:
    """Add the forms a group already implies, and drop the rare ones."""
    source = boolean if isinstance(boolean, str) else ""
    before = parse(source)
    if not before.legal or before.tree is None:
        return EnrichReport(source, source, legal=False)

    raw: list[tuple[tuple[Term, ...], bool]] = []
    _collect(before.tree, False, raw)
    root_is_group = not _has_and(before.tree)

    groups: list[_Group] = []
    for index, (terms, negated) in enumerate(raw):
        group = _Group(terms=terms, negated=negated, index=index)
        group.enrichable = (
            not negated
            and bool(terms)
            and _terms_are_contiguous(source, terms)
            and _closes_safely(source, terms, root_is_group and len(raw) == 1)
        )
        groups.append(group)

    # Plan every edit before writing any of them, so the character budget is
    # spent in one deterministic order rather than group by group.
    for group in groups:
        if not group.enrichable:
            group.kept = list(group.terms)
            continue
        # Additions first: a rare form is only safe to drop once the common
        # forms that replace it are known, and deciding in the other order is
        # what made a second pass find more to do than the first.
        present = {term.normalized for term in group.terms}
        forms, twins = _candidates_for(group, lexicon, present)
        group.added_forms = forms
        group.added_twins = twins
        doomed = _removable_rare(group, lexicon, forms + twins)
        group.kept = [term for term in group.terms if term not in doomed]
        group.removed = [normalize(term.text) for term in doomed]

    def _compose(limits: dict[int, int]) -> str:
        pieces: list[tuple[int, int, str]] = []
        for group in groups:
            if not group.enrichable:
                continue
            allowed = limits.get(group.index, 0)
            additions = (group.added_forms + group.added_twins)[:allowed]
            if not additions and not group.removed:
                continue
            quoted = bool(group.terms[0].quoted)
            rendered = [_render_authored(source, term) for term in group.kept]
            rendered.extend(_render_addition(item, quoted) for item in additions)
            if not rendered:
                continue
            pieces.append(
                (group.terms[0].span[0], group.terms[-1].span[1], " OR ".join(rendered))
            )
        out = source
        for start, end, text in sorted(pieces, key=lambda item: item[0], reverse=True):
            out = out[:start] + text + out[end:]
        return out

    # Removals first at zero additions; they only ever shorten the string.
    limits = {group.index: 0 for group in groups if group.enrichable}
    candidate = _compose(limits)
    truncated = False
    for group in groups:
        if not group.enrichable:
            continue
        total = len(group.added_forms) + len(group.added_twins)
        for step in range(1, total + 1):
            limits[group.index] = step
            grown = _compose(limits)
            if len(grown) > max_chars:
                limits[group.index] = step - 1
                truncated = True
                break
            candidate = grown

    after = parse(candidate)
    dropped = {surface for group in groups for surface in group.removed}
    if not _structure_preserved(before, after, lexicon, dropped):
        return EnrichReport(
            source,
            source,
            legal=True,
            rejected="enrichment would have changed the parse; discarded",
        )

    changes = []
    for group in groups:
        if not group.enrichable:
            continue
        allowed = limits.get(group.index, 0)
        additions = (group.added_forms + group.added_twins)[:allowed]
        forms = tuple(item for item in additions if item in set(group.added_forms))
        twins = tuple(item for item in additions if item in set(group.added_twins))
        change = GroupChange(group.index, forms, twins, tuple(group.removed))
        if not change.empty:
            changes.append(change)

    return EnrichReport(
        source,
        candidate,
        legal=True,
        changes=tuple(changes),
        truncated=truncated,
    )
