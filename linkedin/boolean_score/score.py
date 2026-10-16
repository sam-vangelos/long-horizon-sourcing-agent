"""Scoring one Boolean against one role lexicon (plans/boolean-scorer.md §6).

Six numbers, four of them weighted. Two multiply instead: ``legality``,
because an illegal string scores zero whatever else is true of it, and
``parsimony``, because breadth must never be free. A group holding every form
of every concept is perfect on register, variant and cohesion and worthless as
a search, and only a multiplier says so.

    composite = legality × parsimony × weighted_mean(register_coverage 0.30,
                variant_coverage 0.15, cohesion 0.10, relevance 0.05)

When register or variant coverage has nothing to measure it scores a neutral
0.5 in that mean rather than surrendering its weight, so a string carrying no
lexicon vocabulary is neither punished for coverage it never claimed nor
rewarded for claiming none. Cohesion alone drops out when inapplicable: a
group of one term is not incoherent, it is just short. ``applicable`` reports
the truth in every case.

- ``register_coverage`` (0.30) — per (OR group, concept in it), how much of
  that concept's common register the group carries, plus 0.1 per occasional
  form, capped at 1. This is the reward for writing "evaluation" AND
  "evaluations" AND "eval" AND "evals" instead of one of them.
- ``variant_coverage`` (0.15) — per mapped, non-proper-noun term, how many of
  its twins sit in the same group. Twins are the declared plural/tense forms
  plus the derived hyphen-to-space spelling. Recruiter treats each spelling as
  a separate term, so a missing twin is missing reach. A twin that is itself a
  declared form of the same concept is counted as satisfied here, because
  register coverage already charges its absence and one absence must not be
  charged twice.
- ``relevance`` (0.10) — how close to the role the string can reach. Each
  positive group is worth its most relevant mapped concept (core 1.0,
  adjacent 0.7, peripheral 0.4, nothing mapped 0.2) and the string is worth
  its narrowest group, because an AND only reaches the population its
  tightest gate allows. The relevance of a family is lexicon data.
- ``cohesion`` (0.10) — per OR group with two or more mapped terms, whether
  the group hangs together. An OR group has two legitimate uses. (A)
  exhaustively map the synonyms and analogs of ONE concept. (B) broadly touch
  a set of INTERRELATED concepts — one lexicon family — which is then anchored
  by a second OR group that is more specific and represents a single concept.
  What scores down is a group of UNRELATED concepts, a grab-bag; and, through
  the anchor term in parsimony, a string with no anchor anywhere. Families
  that sit under a shared parent are most of the way to cohesion without
  being the same thing, so a group spanning post-training methods,
  evaluation and data curation scores 0.75 rather than reading as a
  grab-bag. So cohesion is 1 when every mapped term shares one concept
  or one family, and otherwise the share taken by the largest single-family
  cluster.
- ``structure`` (reported, not weighted) — one minus the weight of the
  warning findings. Its charges join the parsimony multiplier instead of the
  weighted mean, because a malformed string must not be able to average its
  defect away against good coverage.
- ``parsimony`` (the multiplier) — ``exp(-1.5 x bloat)``: rare expansions nobody writes,
  groups past eight terms, a concept repeated across sibling AND gates, an
  gate made only of ubiquitous words, an umbrella term OR'd into an
  otherwise specific group (a group is as broad as its broadest term), a NOT
  list past four, a NOT term that excludes the role's own class, any group
  that is a grab-bag, a multi-group string with no single-concept anchor, and
  a length ramp that starts biting at 1,200 characters. A single-group string
  is never charged for a missing anchor, and neither is a string one of whose
  positive groups holds no mapped term at all: a company or title facet may be
  doing the anchoring, and a group the lexicon cannot read is a group it
  cannot call unanchored.

  An anchor is a set of GENERAL terms that directly implicate a certain CLASS
  of thing being searched for — not a single concept, and not the most
  specific thing in the string. A title cluster implicates the class "engineer
  of this kind"; ("LLM" OR "large language model" OR "language models")
  implicates "works on LLMs"; ("evaluation" OR "evals") implicates "does
  evaluation work". Named methods and artifacts — RLHF, DPO, SWE-bench,
  OpenRLHF — are instances rather than classes, and a group of them is a sweep
  through named things, never an anchor. So a positive OR group anchors when
  every concept it maps is a class concept and all of them share at least one
  family, and every instance the group names belongs to that same family;
  unmapped terms are ignored. Naming a few of the things in a class does not
  stop a group implicating the class. When the caller knows a facet filter is
  in play, ``score(..., anchored_by_filter=True)`` drops the term. Likewise
  a long NOT list is bloat by default but a technique when it is deliberate —
  subtract the titles you know to surface the ones you do not — so
  ``score(..., discovery_not=True)`` drops the NOT term and nothing else.

Unmapped terms are ignored by coverage and cohesion, never penalised: the
lexicon's silence is not the string's fault.

The operator's two reference strings, measured against the first role lexicon
over a 76-string corpus whose composite has mean 0.737 and median 0.720:

    ("evaluation" OR "evaluations" OR "eval" OR "evals") AND
    ("agentic coding" OR "software engineering agents" OR "coding agents" OR "code gen")

scores 0.93 — register_coverage 0.875 (the second group is missing the
singular "coding agent"), variant_coverage 0.80 (same absence, charged on the
twin axis), cohesion 1.0, structure 1.0, parsimony 1.0, against a target of
0.9.

    ("RLHF" OR "DPO" OR "SFT") AND ("led" OR "managed")

scores 0.76 against a target of 0.4. The scorer sees what is wrong with it —
parsimony 0.50, charged once for the ubiquitous-verb gate and once for having
no single-concept anchor — but not how wrong. Two structural reasons, both
reported in plans/boolean-scorer.md §9 rather than tuned away here. First,
the lexicon marks acronym expansions ``rare``, so a bare "RLHF" already
carries its whole common register and earns register_coverage 1.0. Second,
none of its terms has a twin, so variant_coverage is inapplicable and its
0.15 is redistributed over the components the string happens to do well on,
lifting the composite from 0.65 to 0.76. A string built only of twin-less
acronyms is therefore scored on a more forgiving denominator than one that
reaches for variants and misses.

Standard library only; nothing here is imported from elsewhere in this repo.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .lexicon import RARE_REGISTER, Concept, Lexicon, normalize
from .parser import And, Finding, Node, Not, Or, ParseError, ParseResult, Term, parse

WEIGHTS: Mapping[str, float] = {
    "register_coverage": 0.30,
    "variant_coverage": 0.15,
    "cohesion": 0.10,
    "relevance": 0.05,
}

STRUCTURE = "structure"

RELEVANCE_SCORES: Mapping[str, float] = {
    "core": 1.0,
    "adjacent": 0.7,
    "peripheral": 0.4,
}
UNMAPPED_RELEVANCE = 0.2

# Coverage that has nothing to measure scores neutral instead of surrendering
# its weight. Cohesion is not in here: a group of one term is short, not
# incoherent, so it drops out of the mean entirely.
NEUTRAL_WHEN_INAPPLICABLE = frozenset({"register_coverage", "variant_coverage"})
# Register coverage is inapplicable only when nothing at all maps, and the
# scorer is not vouching for a string it cannot read.
NEUTRAL_SCORE = 0.20
# Parsimony decays instead of clamping. A clamp makes every string past the
# cap indistinguishable, which leaves a training policy with no gradient
# among bad completions.
PARSIMONY_DECAY = 1.5
LOG_FLOOR = 1e-6
# Every extra real form of a concept already present is worth something. The
# credit does not cap at the concept, only at the composite, because more
# ways of saying a thing people write is strictly better than fewer.
OCCASIONAL_CREDIT = 0.05

# Two legitimate builds: an anchor AND a sweep, or a single exhaustive
# sweep, where a facet outside the keyword field may stand in for the anchor.
# A lone anchor group is neither — general terms with nothing to pin them is
# a wide net — so shape multiplies like parsimony does.
SHAPE_SCORES: Mapping[str, float] = {
    "anchor_and_sweep": 1.00,
    "two_groups_no_anchor": 0.85,
    "anchored_by_filter": 1.00,
    "lone_sweep": 0.90,
    "lone_anchor": 0.75,
}
PARSIMONY = "parsimony"

BLOAT_COEFFICIENTS: Mapping[str, float] = {
    "rare_form": 0.04,
    "repeated_concept": 0.25,
    "generic_and_gate": 0.35,
    "excess_not_term": 0.10,
    "grab_bag": 0.30,
    "umbrella_and_gate": 0.75,
    "employer_gate": 0.50,
    "employer_facet_gate": 0.10,
    "over_gated": 0.40,
    "noisy_term": 0.10,
    "self_exclusion": 0.30,
    "umbrella_in_group": 0.30,
}

NOISY_TERM_CAP = 0.30
RARE_FORM_CAP = 0.30
SELF_EXCLUSION_CAP = 0.75
# A single-family instance sweep is judged on how many of the family's named
# things it reaches, saturating here.
SWEEP_TARGET = 15

GRAB_BAG_THRESHOLD = 0.5
# A group spanning families that share a parent is broad, not incoherent.
PARENT_COHESION = 0.75

STRUCTURE_WEIGHTS: Mapping[str, float] = {
    "unquoted_multiword_run": 0.10,
    "lowercase_operator": 0.20,
    "noop_special_character": 0.20,
    "comma_numeral": 0.10,
    "mid_word_stem": 0.20,
    "nesting_depth_over_2": 0.20,
    "and_groups_over_3": 0.15,
    "sentence_fragment": 0.15,
    "length_over_cap": 0.30,
}

# Codes whose per-occurrence charge stops accumulating.
_STRUCTURE_CAPS = {"unquoted_multiword_run": 0.30, "sentence_fragment": 0.45}

# Findings charged once per occurrence; every other code is charged once.
_PER_OCCURRENCE_CODES = frozenset(
    {
        "unquoted_multiword_run",
        "noop_special_character",
        "and_groups_over_3",
        "sentence_fragment",
    }
)

# Articles and possessives that mark a quoted span as a sentence fragment
# rather than a term. Hyphenated spellings are exempt by construction: the
# check splits on whitespace, so "LLM-as-a-judge" is one token.
_ARTICLE_KEYS = frozenset({"a", "an", "the", "my", "our", "their", "your"})

NOT_TERM_LIMIT = 4
NESTING_LIMIT = 2
AND_GATE_LIMIT = 2
# The Recruiter keyword field's limit as the existing compiler records it,
# unverified, which is why passing it is a finding and not a legality error.
LENGTH_CAP = 2000

_NOOP_SPECIAL_CHARACTERS = frozenset("$%&+")
_BARE_AMPERSAND_RE = re.compile(r"(?<![A-Za-z_])&(?![A-Za-z_])")
_COMMA_NUMERAL_RE = re.compile(r"\d,\d")
_MID_WORD_STEM_SUFFIX_KEYS = ("nc", "iz", "yz")
# Whole words that legitimately end in a stem-suspicious suffix, mirrored from
# the existing compiler's list so the two agree on what reads as truncated.
_MID_WORD_STEM_EXCEPTION_KEYS = frozenset(
    {"zinc", "quiz", "showbiz", "biz", "inc", "sync", "async", "func"}
)

_WARNING = "warning"


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredGroup:
    """One OR group (or a lone term, which is a group of one) and what its
    terms mapped to."""

    terms: tuple[Term, ...]
    negated: bool
    concept_ids: tuple[str | None, ...]

    @property
    def surfaces(self) -> frozenset[str]:
        return frozenset(term.normalized for term in self.terms)

    @property
    def mapped_ids(self) -> tuple[str, ...]:
        return tuple(cid for cid in self.concept_ids if cid)

    def to_dict(self) -> dict:
        return {
            "negated": self.negated,
            "terms": [term.text for term in self.terms],
            "concepts": {
                term.normalized: cid for term, cid in zip(self.terms, self.concept_ids)
            },
        }


def _collect_groups(node: Node, negated: bool, out: list[tuple[tuple[Term, ...], bool]]) -> None:
    if isinstance(node, Term):
        out.append(((node,), negated))
        return
    if isinstance(node, Or):
        terms = tuple(child for child in node.children if isinstance(child, Term))
        if terms:
            out.append((terms, negated))
        for child in node.children:
            if not isinstance(child, Term):
                _collect_groups(child, negated, out)
        return
    if isinstance(node, And):
        for child in node.children:
            _collect_groups(child, negated, out)
        return
    if isinstance(node, Not):
        _collect_groups(node.child, True, out)


def _groups_for(node: Node | None, lexicon: Lexicon) -> tuple[ScoredGroup, ...]:
    if node is None:
        return ()
    raw: list[tuple[tuple[Term, ...], bool]] = []
    _collect_groups(node, False, raw)
    groups: list[ScoredGroup] = []
    for terms, negated in raw:
        ids = tuple(_concept_id(lexicon, term) for term in terms)
        groups.append(ScoredGroup(terms, negated, ids))
    return tuple(groups)


def _concept_id(lexicon: Lexicon, term: Term) -> str | None:
    concept = lexicon.concept_for(term.text)
    return concept.id if concept else None


def _and_nodes(node: Node | None) -> list[And]:
    found: list[And] = []
    if isinstance(node, And):
        found.append(node)
        for child in node.children:
            found.extend(_and_nodes(child))
    elif isinstance(node, Or):
        for child in node.children:
            found.extend(_and_nodes(child))
    elif isinstance(node, Not):
        found.extend(_and_nodes(node.child))
    return found


def _terms_under(node: Node) -> list[Term]:
    if isinstance(node, Term):
        return [node]
    if isinstance(node, (And, Or)):
        collected: list[Term] = []
        for child in node.children:
            collected.extend(_terms_under(child))
        return collected
    if isinstance(node, Not):
        return _terms_under(node.child)
    return []


def _negated_terms(node: Node | None, negated: bool = False) -> list[Term]:
    if node is None:
        return []
    if isinstance(node, Term):
        return [node] if negated else []
    if isinstance(node, (And, Or)):
        collected: list[Term] = []
        for child in node.children:
            collected.extend(_negated_terms(child, negated))
        return collected
    if isinstance(node, Not):
        return _negated_terms(node.child, True)
    return []


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# A class concept says what kind of person the string is looking for, so it
# carries twice the weight of a named instance in the register mean.
def _sweep_family(group: ScoredGroup, lexicon: Lexicon) -> str | None:
    """The family this group sweeps, when it is a pure sweep of named things.

    Two or more instance concepts, no class concept, and one family shared by
    all of them. A sweep is judged on how much of that family it reaches, not
    on the surface forms of names that only have one.
    """
    concepts = [
        concept
        for concept in (
            _concept_by_id(lexicon, concept_id) for concept_id in set(group.mapped_ids)
        )
        if concept is not None
    ]
    if len(concepts) < 2 or any(concept.names_a_class for concept in concepts):
        return None
    shared = set(concepts[0].families)
    for concept in concepts[1:]:
        shared &= set(concept.families)
    if not shared:
        return None
    return max(shared, key=lambda family: (lexicon.instances_in_family(family), family))


def _register_coverage(groups: Iterable[ScoredGroup], lexicon: Lexicon) -> float | None:
    """How much of each concept's register the group carries.

    Each concept counts for as many slots as it owes common forms, so one form
    present out of four cannot be averaged away by three concepts owing one
    form each. A pure sweep of named things is scored differently: a name with
    one spelling has no register to cover, so the group is judged on how much
    of its family it reaches.
    """
    scores: list[tuple[float, float]] = []
    for group in groups:
        present = group.surfaces
        sweep = _sweep_family(group, lexicon)
        if sweep is not None:
            reached = len(set(group.mapped_ids))
            known = min(SWEEP_TARGET, max(1, lexicon.instances_in_family(sweep)))
            scores.append((min(1.0, reached / known), float(reached)))
            continue
        seen: dict[str, Concept] = {}
        for term in group.terms:
            concept = lexicon.concept_for(term.text)
            if concept is not None:
                seen.setdefault(concept.id, concept)
        for concept in seen.values():
            expected = lexicon.expected_forms(concept)
            occasional = sum(
                1 for form in concept.occasional_forms if form.normalized in present
            )
            if not expected:
                # Vocabulary the lexicon says nobody writes: the occasional
                # forms are all there is to earn.
                scores.append((min(1.0, 0.1 * occasional), 1.0))
                continue
            hits = sum(1 for form in expected if form.normalized in present)
            # More real forms of a concept already present is strictly better,
            # never neutral, so the occasional credit is uncapped here; the
            # composite clamps at 1.0, not the concept.
            fraction = min(1.0, hits / len(expected)) + OCCASIONAL_CREDIT * occasional
            scores.append((fraction, float(len(expected))))
    if not scores:
        return None
    total = sum(weight for _value, weight in scores)
    return min(1.0, sum(value * weight for value, weight in scores) / total)


def _variant_coverage(groups: Iterable[ScoredGroup], lexicon: Lexicon) -> float | None:
    scores: list[float] = []
    for group in groups:
        present = group.surfaces
        for term in group.terms:
            concept = lexicon.concept_for(term.text)
            if concept is None or concept.proper_noun:
                continue
            twins = lexicon.twins_for(term.text)
            if not twins:
                continue
            satisfied = sum(
                1
                for twin in twins
                if twin in present or lexicon.declares_form(concept, twin)
            )
            scores.append(satisfied / len(twins))
    return _mean(scores)


def _group_cohesion(group: ScoredGroup, lexicon: Lexicon) -> float | None:
    """How well one group hangs together, or ``None`` when it is too short.

    A group hangs together when every mapped term names the same concept, or
    when some one family is shared by all of them. Otherwise it scores the
    largest cluster that does share a family.
    """
    mapped = group.mapped_ids
    if len(mapped) < 2:
        return None
    if len(set(mapped)) == 1:
        return 1.0
    per_term = [_families_of(lexicon, concept_id) for concept_id in mapped]
    if set(per_term[0]).intersection(*(set(item) for item in per_term[1:])):
        return 1.0
    reach = [lexicon.family_reach(concept_id) for concept_id in mapped]
    if set(reach[0]).intersection(*(set(item) for item in reach[1:])):
        return PARENT_COHESION
    counts: Counter[str] = Counter()
    for families in reach:
        counts.update(set(families))
    return (max(counts.values()) if counts else 1) / len(mapped)


def _cohesion(groups: Iterable[ScoredGroup], lexicon: Lexicon) -> float | None:
    scores = [
        value
        for value in (_group_cohesion(group, lexicon) for group in groups)
        if value is not None
    ]
    return _mean(scores)


def _families_of(lexicon: Lexicon, concept_id: str) -> tuple[str, ...]:
    for concept in lexicon.concepts:
        if concept.id == concept_id:
            return concept.families
    return (concept_id,)


def _is_anchor(group: ScoredGroup, lexicon: Lexicon) -> bool:
    """A group of GENERAL terms implicating one class of candidate.

    Its class concepts must share a family or a parent — the same reach
    cohesion uses, because the anchor people actually write spans two
    families, "LLM" from the umbrella beside "fine-tuning" from the methods —
    and every instance the group names must share that reach. A group with no
    class concept is a sweep through named things, unless its family
    implicates a class of person by itself: a conference list anchors
    published researchers.
    """
    mapped = set(group.mapped_ids)
    if not mapped:
        return False
    concepts = [_concept_by_id(lexicon, concept_id) for concept_id in mapped]
    if any(concept is None for concept in concepts):
        return False
    classes = [concept for concept in concepts if concept.names_a_class]
    instances = [concept for concept in concepts if not concept.names_a_class]
    if not classes:
        return bool(instances) and all(
            lexicon.anchors_by_family(concept) for concept in instances
        )
    shared = set(lexicon.family_reach(classes[0].id))
    for concept in classes[1:]:
        shared &= set(lexicon.family_reach(concept.id))
    return any(
        all(family in lexicon.family_reach(concept.id) for concept in instances)
        for family in shared
    )


def _concept_by_id(lexicon: Lexicon, concept_id: str) -> Concept | None:
    for concept in lexicon.concepts:
        if concept.id == concept_id:
            return concept
    return None


def _has_discriminator(groups: Iterable[ScoredGroup], lexicon: Lexicon) -> bool:
    """True when some mapped term in a POSITIVE group names more than the
    field itself. What a string rules out cannot narrow what it rules in, so
    a NOT list never supplies the discriminator."""
    return any(
        lexicon.concept_for(term.text) is not None and not lexicon.is_umbrella(term.text)
        for group in groups
        if not group.negated
        for term in group.terms
    )


def _wasted_gates(
    groups: Sequence[ScoredGroup], lexicon: Lexicon
) -> tuple[int, int, int, int]:
    """(verb gates, umbrella gates, employer gates, employer facet gates).

    A group of ubiquitous VERBS is a wasted gate however good the rest of the
    string is. A group of UMBRELLA terms — the field's own name — is a wasted
    gate only when nothing else in the string narrows it, which is the
    discriminator rule the existing compiler applies to bare field names, and
    only positive groups can supply that discriminator. A group of EMPLOYER
    names is a filter the company facet should have carried, and a filter
    standing in for a search entirely when nothing narrows it at all.
    """
    discriminator = _has_discriminator(groups, lexicon)
    verbs = umbrella = employers = facets = 0
    for group in groups:
        if group.negated or not group.terms:
            continue
        if all(lexicon.is_employer(term.text) for term in group.terms):
            if discriminator:
                facets += 1
            else:
                employers += 1
            continue
        if all(lexicon.is_generic_verb(term.text) for term in group.terms):
            verbs += 1
        elif not discriminator and all(
            lexicon.is_ubiquitous(term.text) for term in group.terms
        ):
            umbrella += 1
    return verbs, umbrella, employers, facets


def shape_of(
    groups: Sequence[ScoredGroup], lexicon: Lexicon, *, anchored_by_filter: bool
) -> str:
    """Which of the two legitimate builds this string is, or neither."""
    positive = [group for group in groups if not group.negated]
    if len(positive) >= 2:
        return (
            "anchor_and_sweep"
            if any(_is_anchor(group, lexicon) for group in positive)
            else "two_groups_no_anchor"
        )
    if anchored_by_filter:
        return "anchored_by_filter"
    if not positive:
        return "lone_sweep"
    mapped = set(positive[0].mapped_ids)
    if not mapped:
        return "lone_sweep"
    concepts = [_concept_by_id(lexicon, concept_id) for concept_id in mapped]
    if any(
        concept is not None and not concept.names_a_class for concept in concepts
    ):
        return "lone_sweep"
    return "lone_anchor"


def _relevance(groups: Sequence[ScoredGroup], lexicon: Lexicon) -> float:
    """How close to the role the string can reach.

    Each positive group is worth its most relevant mapped concept, and the
    string is worth its narrowest group: an AND only reaches the population
    its tightest gate allows.
    """
    per_group: list[float] = []
    for group in groups:
        if group.negated or not group.mapped_ids:
            # A group the lexicon cannot read does not set the population's
            # topic; an employer list says who, not what.
            continue
        best = UNMAPPED_RELEVANCE
        for concept_id in set(group.mapped_ids):
            concept = _concept_by_id(lexicon, concept_id)
            if concept is None:
                continue
            best = max(best, RELEVANCE_SCORES[lexicon.relevance_of(concept)])
        per_group.append(best)
    return min(per_group) if per_group else UNMAPPED_RELEVANCE


def _parsimony(
    boolean: str,
    tree: Node | None,
    groups: Sequence[ScoredGroup],
    lexicon: Lexicon,
    *,
    structure_charges: Mapping[str, float],
    anchor_present: bool,
    anchored_by_filter: bool,
    discovery_not: bool,
    grab_bag_groups: Sequence[ScoredGroup],
) -> tuple[float, dict[str, float]]:
    rare_forms = 0
    for group in groups:
        for term in group.terms:
            form = lexicon.form_for(term.text)
            if form is not None and form.register == RARE_REGISTER:
                rare_forms += 1

    self_excluded = sum(
        1
        for term in _negated_terms(tree)
        if (concept := lexicon.concept_for(term.text)) is not None
        and concept.names_a_class
    )

    swallowed = 0
    swallowed_groups: set[int] = set()
    for group in groups:
        if group.negated:
            continue
        # An anchor group is general BY DESIGN: naming the class and a few of
        # its instances is the shape the anchor rule asks for, so the
        # swallowed-group charge would contradict it.
        umbrella_here = not _is_anchor(group, lexicon) and any(
            lexicon.is_umbrella(term.text) for term in group.terms
        )
        specific_here = any(
            lexicon.concept_for(term.text) is not None
            and not lexicon.is_umbrella(term.text)
            for term in group.terms
        )
        if umbrella_here and specific_here:
            swallowed += 1
            swallowed_groups.add(id(group))

    repeated = 0
    for node in _and_nodes(tree):
        operands = [child for child in node.children if not isinstance(child, Not)]
        tally: Counter[str] = Counter()
        for operand in operands:
            tally.update(
                {
                    cid
                    for cid in (
                        _concept_id(lexicon, term) for term in _terms_under(operand)
                    )
                    if cid is not None
                }
            )
        repeated += sum(count - 1 for count in tally.values() if count > 1)

    verb_gates, umbrella_gates, employer_gates, facet_gates = _wasted_gates(
        groups, lexicon
    )
    noisy = sum(
        1
        for group in groups
        for term in group.terms
        if lexicon.is_noisy(term.text)
    )
    over_gated = max(
        0,
        max(
            (
                len([child for child in node.children if not isinstance(child, Not)])
                for node in _and_nodes(tree)
            ),
            default=0,
        )
        - AND_GATE_LIMIT,
    )

    excess_not = 0 if discovery_not else max(0, len(_negated_terms(tree)) - NOT_TERM_LIMIT)

    detail = {
        "rare_form": min(
            RARE_FORM_CAP, BLOAT_COEFFICIENTS["rare_form"] * rare_forms
        ),
        "self_exclusion": min(
            SELF_EXCLUSION_CAP,
            BLOAT_COEFFICIENTS["self_exclusion"] * self_excluded,
        ),
        "umbrella_in_group": BLOAT_COEFFICIENTS["umbrella_in_group"] * swallowed,
        "repeated_concept": BLOAT_COEFFICIENTS["repeated_concept"] * repeated,
        "generic_and_gate": BLOAT_COEFFICIENTS["generic_and_gate"] * verb_gates,
        "umbrella_and_gate": BLOAT_COEFFICIENTS["umbrella_and_gate"] * umbrella_gates,
        "employer_gate": BLOAT_COEFFICIENTS["employer_gate"] * employer_gates,
        "employer_facet_gate": (
            BLOAT_COEFFICIENTS["employer_facet_gate"] * facet_gates
        ),
        "over_gated": BLOAT_COEFFICIENTS["over_gated"] * over_gated,
        "noisy_term": min(
            NOISY_TERM_CAP, BLOAT_COEFFICIENTS["noisy_term"] * noisy
        ),
        "excess_not_term": BLOAT_COEFFICIENTS["excess_not_term"] * excess_not,
        "grab_bag": (
            BLOAT_COEFFICIENTS["grab_bag"]
            if any(id(group) not in swallowed_groups for group in grab_bag_groups)
            else 0.0
        ),
    }
    # A malformed string must not be able to average its defect away against
    # good coverage, so the structure charges multiply with everything else.
    detail.update(structure_charges)
    bloat = sum(detail.values())
    return math.exp(-PARSIMONY_DECAY * bloat), detail


# ---------------------------------------------------------------------------
# Structure findings the scorer adds to the parser's
# ---------------------------------------------------------------------------


def _and_twin(term: Term) -> str:
    """The spelled-out twin of an ampersand phrase, or "" when there is none.

    Mirrors the compiler: "R&D" expects "R and D" beside it, and a group that
    carries both has not lost the term to the tokenizer.
    """
    if not _BARE_AMPERSAND_RE.search(term.text):
        return ""
    return normalize(_BARE_AMPERSAND_RE.sub(" and ", term.text))


def _is_sentence_fragment(term: Term) -> bool:
    """A quoted span carrying a bare article or possessive reads as prose."""
    return any(token.lower() in _ARTICLE_KEYS for token in term.text.split())


def _ends_mid_word(term: Term) -> bool:
    tokens = term.text.split()
    if not tokens:
        return False
    last = tokens[-1]
    lowered = last.lower()
    if last.endswith("-"):
        return True
    if lowered in _MID_WORD_STEM_EXCEPTION_KEYS:
        return False
    return any(lowered.endswith(suffix) for suffix in _MID_WORD_STEM_SUFFIX_KEYS)


def _structure_findings(result: ParseResult, groups: Sequence[ScoredGroup]) -> list[Finding]:
    findings: list[Finding] = []
    for group in groups:
        for term in group.terms:
            if not term.quoted:
                continue
            specials = {
                char for char in set(term.text) if char in _NOOP_SPECIAL_CHARACTERS
            }
            twin = _and_twin(term)
            if twin and twin in group.surfaces:
                specials.discard("&")
            if specials:
                findings.append(
                    Finding(
                        "noop_special_character",
                        _WARNING,
                        f"{term.text!r} carries a character the tokenizer drops, "
                        "so it will not match as written.",
                        term.span,
                    )
                )
            if _COMMA_NUMERAL_RE.search(term.text):
                findings.append(
                    Finding(
                        "comma_numeral",
                        _WARNING,
                        f"{term.text!r} carries a comma-separated numeral, which "
                        "tokenizes on the comma.",
                        term.span,
                    )
                )
            if _is_sentence_fragment(term):
                findings.append(
                    Finding(
                        "sentence_fragment",
                        _WARNING,
                        f"{term.text!r} reads as a sentence rather than a term; "
                        "the article will not help it match.",
                        term.span,
                    )
                )
            if _ends_mid_word(term):
                findings.append(
                    Finding(
                        "mid_word_stem",
                        _WARNING,
                        f"{term.text!r} appears to end mid-word; a stem fragment "
                        "does not match as a whole term.",
                        term.span,
                    )
                )
    if len(result.source) > LENGTH_CAP:
        findings.append(
            Finding(
                "length_over_cap",
                _WARNING,
                f"{len(result.source)} characters is past the {LENGTH_CAP}-character "
                "keyword field; the tail may be dropped.",
                None,
            )
        )
    if result.max_depth > NESTING_LIMIT:
        findings.append(
            Finding(
                "nesting_depth_over_2",
                _WARNING,
                f"Nesting reaches depth {result.max_depth}; Recruiter strings read "
                "flat in practice.",
                None,
            )
        )
    for node in _and_nodes(result.tree):
        gates = len([child for child in node.children if not isinstance(child, Not)])
        for _ in range(max(0, gates - AND_GATE_LIMIT)):
            findings.append(
                Finding(
                    "and_groups_over_3",
                    _WARNING,
                    f"{gates} groups joined by AND; each extra gate narrows the "
                    "result set multiplicatively.",
                    None,
                )
            )
    return findings


def _structure(findings: Sequence[Finding]) -> tuple[float, dict[str, float]]:
    counts = Counter(finding.code for finding in findings)
    detail: dict[str, float] = {}
    for code, count in counts.items():
        weight = STRUCTURE_WEIGHTS.get(code)
        if weight is None:
            continue
        charged = weight * count if code in _PER_OCCURRENCE_CODES else weight
        cap = _STRUCTURE_CAPS.get(code)
        if cap is not None:
            charged = min(charged, cap)
        detail[code] = charged
    return 1.0 - min(1.0, sum(detail.values())), detail


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreReport:
    boolean: str
    role: str
    legal: bool
    composite: float
    components: Mapping[str, float | None]
    applicable: Mapping[str, bool]
    weights: Mapping[str, float]
    findings: tuple[Finding, ...]
    errors: tuple[ParseError, ...]
    groups: tuple[ScoredGroup, ...]
    shape: str = "anchor_and_sweep"
    anchor_present: bool = False
    anchored_by_filter: bool = False
    discovery_not: bool = False
    bloat: Mapping[str, float] = field(default_factory=dict)
    structure_charges: Mapping[str, float] = field(default_factory=dict)

    @property
    def log_composite(self) -> float:
        """The composite in log space.

        A multiplicative reward makes any improvement on a bad completion
        invisible in absolute terms, and GRPO normalises advantages within a
        group, so the training rig should consume this rather than the raw
        composite.
        """
        return math.log(max(self.composite, LOG_FLOOR))

    def to_dict(self) -> dict:
        return {
            "boolean": self.boolean,
            "role": self.role,
            "legal": self.legal,
            "composite": self.composite,
            "components": dict(self.components),
            "applicable": dict(self.applicable),
            "weights": dict(self.weights),
            "findings": [finding.to_dict() for finding in self.findings],
            "errors": [error.to_dict() for error in self.errors],
            "groups": [group.to_dict() for group in self.groups],
            "shape": self.shape,
            "shape_multiplier": SHAPE_SCORES.get(self.shape, 1.0),
            "log_composite": self.log_composite,
            "anchor_present": self.anchor_present,
            "anchored_by_filter": self.anchored_by_filter,
            "discovery_not": self.discovery_not,
            "bloat": dict(self.bloat),
            "structure_charges": dict(self.structure_charges),
        }


def _illegal_report(
    boolean: str,
    lexicon: Lexicon,
    result: ParseResult,
    anchored_by_filter: bool,
    discovery_not: bool,
) -> ScoreReport:
    components: dict[str, float | None] = {"legality": 0.0}
    applicable: dict[str, bool] = {"legality": True}
    for name in (*WEIGHTS, STRUCTURE, PARSIMONY):
        components[name] = None
        applicable[name] = False
    return ScoreReport(
        boolean=boolean,
        role=lexicon.role,
        legal=False,
        composite=0.0,
        components=components,
        applicable=applicable,
        weights=dict(WEIGHTS),
        findings=result.findings,
        errors=result.errors,
        groups=(),
        anchored_by_filter=anchored_by_filter,
        discovery_not=discovery_not,
    )


def score(
    boolean: str,
    lexicon: Lexicon,
    *,
    anchored_by_filter: bool = False,
    discovery_not: bool = False,
) -> ScoreReport:
    """Score one Boolean string against one role lexicon.

    ``anchored_by_filter`` tells the scorer that a facet outside the keyword
    field — a company list, a title filter — is doing the anchoring, so the
    missing-anchor bloat term does not apply. ``discovery_not`` tells it the
    long NOT list is deliberate, so the excess-NOT bloat term does not apply.
    Both default to false, so an unannotated string is judged on what it says.
    """
    result = parse(boolean)
    if not result.legal:
        return _illegal_report(
            boolean, lexicon, result, anchored_by_filter, discovery_not
        )

    groups = _groups_for(result.tree, lexicon)
    positive = [group for group in groups if not group.negated]
    anchor_present = any(_is_anchor(group, lexicon) for group in positive)
    grab_bag_groups = [
        group
        for group in positive
        if (value := _group_cohesion(group, lexicon)) is not None
        and value < GRAB_BAG_THRESHOLD
    ]
    findings = list(result.findings) + _structure_findings(result, groups)
    structure, structure_charges = _structure(findings)
    shape = shape_of(groups, lexicon, anchored_by_filter=anchored_by_filter)
    parsimony, bloat = _parsimony(
        boolean,
        result.tree,
        groups,
        lexicon,
        structure_charges=structure_charges,
        anchor_present=anchor_present,
        anchored_by_filter=anchored_by_filter,
        discovery_not=discovery_not,
        grab_bag_groups=grab_bag_groups,
    )

    components: dict[str, float | None] = {
        "legality": 1.0,
        "register_coverage": _register_coverage(groups, lexicon),
        "variant_coverage": _variant_coverage(groups, lexicon),
        "cohesion": _cohesion(groups, lexicon),
        STRUCTURE: structure,
        "relevance": _relevance(groups, lexicon),
        PARSIMONY: parsimony,
    }
    applicable = {name: components[name] is not None for name in components}

    numerator = 0.0
    denominator = 0.0
    for name, weight in WEIGHTS.items():
        value = components[name]
        if value is None:
            if name not in NEUTRAL_WHEN_INAPPLICABLE:
                continue
            # Variant coverage is inapplicable on plenty of good strings, so
            # its neutral is what the population actually averages; register
            # coverage is inapplicable only when nothing maps at all.
            value = (
                lexicon.neutral_variant
                if name == "variant_coverage"
                else NEUTRAL_SCORE
            )
        numerator += weight * value
        denominator += weight
    composite = (
        parsimony * SHAPE_SCORES[shape] * (numerator / denominator)
        if denominator
        else 0.0
    )

    return ScoreReport(
        boolean=boolean,
        role=lexicon.role,
        legal=True,
        composite=composite,
        components=components,
        applicable=applicable,
        weights=dict(WEIGHTS),
        findings=tuple(findings),
        errors=(),
        groups=groups,
        shape=shape,
        anchor_present=anchor_present,
        anchored_by_filter=anchored_by_filter,
        discovery_not=discovery_not,
        bloat=bloat,
        structure_charges=structure_charges,
    )
