"""Role lexicon: the data the scorer reads a Boolean against
(plans/boolean-scorer.md §5).

Every word the scorer knows lives in a JSON file, never in this module. That
is the repo's vertical-vocabulary ratchet stated as a design rule, and it is
also what makes the package portable: a second role is a second JSON file, not
a second code path.

A lexicon declares concepts; a concept declares whether it names a CLASS of
thing or an INSTANCE of one, declares one or more families, and declares its
surface forms; a form declares a ``register`` (how routinely a person writes
it) and a ``kind`` (what kind of surface it is). The families are what tell
the scorer which concepts are interrelated, so that a group reaching across
one family reads as deliberate breadth and a group reaching across four reads
as a grab-bag. A concept may sit in more than one — GRPO is a post-training
method and an RL-environment term at once — which is why ``family`` accepts a
string or a list, ``families`` is accepted as its alias, and both normalize to
a tuple. Families may themselves sit under a parent, declared in the optional
top-level ``family_parents``, so that post-training methods, evaluation, data
curation and RL environments can read as one practice without collapsing into
one family.

The class/instance distinction is what lets the scorer tell an anchor from a
sweep. A class concept names a kind of thing a person does or works on — a
title, "LLM", "evaluation" — and a group of them implicates a class of
candidate. An instance concept names one method or one artifact — RLHF, DPO,
SWE-bench — and a group of those is a sweep through named things. A
``proper_noun`` concept is an instance by definition, and the loader refuses a
file that says otherwise.

Ubiquitous vocabulary comes in two registers, and they are charged
differently. ``generic_verbs`` are the words every profile carries whatever
the role ("led", "managed"), and a group made only of them is a wasted gate
however good the rest of the string is. ``umbrella_terms`` name the field
rather than the work, and a group made only of them is a wasted gate only
when nothing else in the string narrows it — the discriminator rule the
existing compiler applies to bare field names. ``employer_terms`` are company,
lab and consultancy names: the company facet owns them, so a group made only
of them is a filter wearing a search's clothes. ``noisy_terms`` are surfaces
whose dominant meaning on LinkedIn is something other than the one intended,
and every one of them is a judgment call the reviewer should see. ``common`` forms are the ones a well-expanded group is
expected to carry, ``occasional`` forms earn a small bonus and are never
required, ``rare`` forms earn nothing and count against parsimony.

Two twin axes sit beside the forms. ``morphology`` declares twins expected in
the same group (a plural, a tense form), symmetrically: declaring one
direction declares both. The hyphen axis is derived rather than declared —
an interior hyphen implies the spaced spelling — and mirrors the existing
compiler's surface-variant helper, including its exemptions: a term carrying a
digit, ``@``, ``+`` or ``#`` is a named artifact whose punctuation is part of
the name, and the space-to-hyphen direction is never synthesised.

A surface form belongs to exactly one concept; the loader refuses a file that
breaks that, refuses an unknown register or kind, and refuses a concept with
no forms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

LEXICON_DIR = Path(__file__).resolve().parent / "lexicons"

COMMON_REGISTER = "common"
OCCASIONAL_REGISTER = "occasional"
RARE_REGISTER = "rare"
REGISTER_KEYS = frozenset({COMMON_REGISTER, OCCASIONAL_REGISTER, RARE_REGISTER})

# Surface kinds are wire-format labels for the reviewer, not vocabulary.
KIND_KEYS = frozenset(
    {
        "acronym",
        "expansion",
        "noun",
        "noun_plural",
        "colloquial",
        "verb_past",
        "verb_gerund",
        "verb_passive",
        "compound",
        "other",
    }
)

CORE_RELEVANCE = "core"
ADJACENT_RELEVANCE = "adjacent"
PERIPHERAL_RELEVANCE = "peripheral"
RELEVANCE_KEYS = frozenset({CORE_RELEVANCE, ADJACENT_RELEVANCE, PERIPHERAL_RELEVANCE})

CLASS_KIND = "class"
INSTANCE_KIND = "instance"
CONCEPT_KIND_KEYS = frozenset({CLASS_KIND, INSTANCE_KIND})

_NAMED_ARTIFACT_RE = re.compile(r"[0-9@+#]")


class LexiconError(ValueError):
    """A lexicon file the loader refuses to trust."""


def normalize(term: str) -> str:
    """Lowercase, collapse whitespace. The one lookup key the scorer uses."""
    return " ".join(str(term).lower().split())


def derived_hyphen_twin(term: str) -> str | None:
    """The spaced spelling implied by an interior hyphen, or ``None``.

    Mirrors the compiler's surface-variant derivation: one orthographic axis,
    hyphen to space and never the reverse; named artifacts carrying a digit or
    operator punctuation are refused; an edge hyphen is not a separator.
    """
    base = normalize(term)
    if not base or _NAMED_ARTIFACT_RE.search(base) or '"' in base:
        return None
    if base.startswith("-") or base.endswith("-"):
        return None
    if "-" not in base:
        return None
    spaced = normalize(base.replace("-", " "))
    return spaced if spaced and spaced != base else None


@dataclass(frozen=True)
class Form:
    text: str
    register: str
    kind: str
    unsure: bool = False

    @property
    def normalized(self) -> str:
        return normalize(self.text)


@dataclass(frozen=True)
class Concept:
    id: str
    families: tuple[str, ...]
    kind: str
    forms: tuple[Form, ...]
    proper_noun: bool = False
    unsure: bool = False
    relevance: str | None = None

    @property
    def names_a_class(self) -> bool:
        return self.kind == CLASS_KIND

    def _by_register(self, register: str) -> tuple[Form, ...]:
        return tuple(form for form in self.forms if form.register == register)

    @property
    def common_forms(self) -> tuple[Form, ...]:
        return self._by_register(COMMON_REGISTER)

    @property
    def occasional_forms(self) -> tuple[Form, ...]:
        return self._by_register(OCCASIONAL_REGISTER)

    @property
    def rare_forms(self) -> tuple[Form, ...]:
        return self._by_register(RARE_REGISTER)


@dataclass(frozen=True)
class Lexicon:
    role: str
    version: int
    concepts: tuple[Concept, ...]
    morphology: Mapping[str, frozenset[str]]
    generic_verbs: frozenset[str]
    umbrella_terms: frozenset[str]
    employer_terms: frozenset[str]
    noisy_terms: frozenset[str]
    family_relevance: Mapping[str, str]
    # The population mean of applicable variant coverage. A neutral fill above
    # the mean is a reward for saying less, so this is what an inapplicable
    # variant score is worth rather than a flat 0.5.
    neutral_variant: float
    anchor_families: frozenset[str]
    family_parents: Mapping[str, str]
    note: str = ""
    _form_index: Mapping[str, tuple[Concept, Form]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # -- lookup
    def concept_for(self, term: str) -> Concept | None:
        found = self._form_index.get(normalize(term))
        return found[0] if found else None

    def form_for(self, term: str) -> Form | None:
        found = self._form_index.get(normalize(term))
        return found[1] if found else None

    def families_for(self, term: str) -> tuple[str, ...]:
        concept = self.concept_for(term)
        return concept.families if concept else ()

    @property
    def family_names(self) -> frozenset[str]:
        return frozenset(
            family for concept in self.concepts for family in concept.families
        )

    def declares_form(self, concept: Concept, term: str) -> bool:
        """True when ``term`` is one of this concept's own declared forms.

        The register axis already measures those, so the twin axis must not
        charge the same absence a second time.
        """
        key = normalize(term)
        return any(form.normalized == key for form in concept.forms)

    def expected_forms(self, concept: Concept) -> tuple[Form, ...]:
        """The forms a well-expanded group is expected to carry."""
        return concept.common_forms

    def is_generic_verb(self, term: str) -> bool:
        return normalize(term) in self.generic_verbs

    def is_umbrella(self, term: str) -> bool:
        return normalize(term) in self.umbrella_terms

    def is_ubiquitous(self, term: str) -> bool:
        return self.is_generic_verb(term) or self.is_umbrella(term)

    def is_employer(self, term: str) -> bool:
        return normalize(term) in self.employer_terms

    def is_noisy(self, term: str) -> bool:
        return normalize(term) in self.noisy_terms

    def relevance_of(self, concept: Concept) -> str:
        """How close this concept sits to the role the lexicon is for.

        A per-concept ``relevance`` wins; otherwise the most generous of its
        families' declared relevances, and a family nobody classified is
        adjacent.
        """
        if concept.relevance:
            return concept.relevance
        ranked = [
            self.family_relevance.get(family, ADJACENT_RELEVANCE)
            for family in concept.families
        ]
        for level in (CORE_RELEVANCE, ADJACENT_RELEVANCE, PERIPHERAL_RELEVANCE):
            if level in ranked:
                return level
        return ADJACENT_RELEVANCE

    def anchors_by_family(self, concept: Concept) -> bool:
        """True when this concept's family implicates a class of person on
        its own — a conference list anchors published researchers even though
        every term in it is a named instance."""
        return any(family in self.anchor_families for family in concept.families)

    def instances_in_family(self, family: str) -> int:
        """How many named instances of this family the lexicon knows.

        The denominator for a sweep: two of thirty coding benchmarks is a
        thin sweep, twenty is a thorough one.
        """
        return sum(
            1
            for concept in self.concepts
            if not concept.names_a_class and family in concept.families
        )

    def parent_of(self, family: str) -> str | None:
        return self.family_parents.get(family)

    def family_reach(self, concept_id: str) -> frozenset[str]:
        """A concept's families, plus whatever parents they sit under."""
        for concept in self.concepts:
            if concept.id == concept_id:
                reach = set(concept.families)
                reach.update(
                    parent
                    for family in concept.families
                    if (parent := self.parent_of(family)) is not None
                )
                return frozenset(reach)
        return frozenset({concept_id})

    def twins_for(self, term: str) -> frozenset[str]:
        """Normalized twins expected alongside ``term`` in the same group."""
        key = normalize(term)
        twins = set(self.morphology.get(key, ()))
        hyphenated = derived_hyphen_twin(key)
        if hyphenated:
            twins.add(hyphenated)
        twins.discard(key)
        return frozenset(twins)

    def unsure_items(self) -> list[str]:
        """Every concept and form the drafter flagged as not confident."""
        flagged: list[str] = []
        for concept in self.concepts:
            if concept.unsure:
                flagged.append(concept.id)
            for form in concept.forms:
                if form.unsure:
                    flagged.append(f"{concept.id}:{form.text}")
        flagged.extend(f"noisy:{term}" for term in sorted(self.noisy_terms))
        return flagged


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _require_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise LexiconError(f"{label} must be an object.")
    return payload


def _load_form(raw: Any, concept_id: str) -> Form:
    data = _require_mapping(raw, f"concept {concept_id!r} form")
    surface = str(data.get("text", "")).strip()
    if not surface:
        raise LexiconError(f"concept {concept_id!r} has a form with no text.")
    register = str(data.get("register", COMMON_REGISTER))
    if register not in REGISTER_KEYS:
        raise LexiconError(
            f"concept {concept_id!r} form {surface!r} declares an unknown register {register!r}."
        )
    kind = str(data.get("kind", "other"))
    if kind not in KIND_KEYS:
        raise LexiconError(
            f"concept {concept_id!r} form {surface!r} declares an unknown kind {kind!r}."
        )
    return Form(surface, register, kind, bool(data.get("unsure", False)))


def _load_concept(raw: Any, seen_ids: set[str]) -> Concept:
    data = _require_mapping(raw, "concept")
    concept_id = str(data.get("id", "")).strip()
    if not concept_id:
        raise LexiconError("a concept declares no id.")
    if concept_id in seen_ids:
        raise LexiconError(f"concept id {concept_id!r} is declared twice.")
    seen_ids.add(concept_id)
    families = _load_families(data, concept_id)
    proper_noun = bool(data.get("proper_noun", False))
    kind = str(data.get("kind", INSTANCE_KIND if proper_noun else "")).strip()
    if kind not in CONCEPT_KIND_KEYS:
        raise LexiconError(
            f"concept {concept_id!r} must declare kind as {CLASS_KIND!r} or "
            f"{INSTANCE_KIND!r}."
        )
    if proper_noun and kind != INSTANCE_KIND:
        raise LexiconError(
            f"concept {concept_id!r} is a proper noun, so it names an instance, "
            f"not a {CLASS_KIND}."
        )
    raw_forms = data.get("forms") or []
    if not isinstance(raw_forms, list) or not raw_forms:
        raise LexiconError(f"concept {concept_id!r} declares no forms.")
    forms = tuple(_load_form(item, concept_id) for item in raw_forms)
    if not any(form.register in (COMMON_REGISTER, OCCASIONAL_REGISTER) for form in forms):
        raise LexiconError(
            f"concept {concept_id!r} declares nothing but rare forms."
        )
    relevance = data.get("relevance")
    if relevance is not None:
        relevance = str(relevance).strip()
        if relevance not in RELEVANCE_KEYS:
            raise LexiconError(
                f"concept {concept_id!r} declares an unknown relevance {relevance!r}."
            )
    return Concept(
        id=concept_id,
        families=families,
        kind=kind,
        forms=forms,
        proper_noun=proper_noun,
        unsure=bool(data.get("unsure", False)),
        relevance=relevance,
    )


def _load_families(data: Mapping[str, Any], concept_id: str) -> tuple[str, ...]:
    """One family or several, declared as ``family`` or as ``families``."""
    declared = data.get("families", data.get("family"))
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list) or not declared:
        raise LexiconError(f"concept {concept_id!r} declares no family.")
    families = tuple(dict.fromkeys(str(item).strip() for item in declared))
    if not all(families):
        raise LexiconError(f"concept {concept_id!r} declares an empty family.")
    return families


def _load_morphology(raw: Any, known: Iterable[str]) -> dict[str, frozenset[str]]:
    """Symmetric closure of the declared twin pairs, normalized."""
    if raw is None:
        raw = {}
    data = _require_mapping(raw, "morphology")
    groups: list[set[str]] = []
    for key, values in data.items():
        if not isinstance(values, list) or not values:
            raise LexiconError(f"morphology entry {key!r} declares no twins.")
        cluster = {normalize(key)} | {normalize(value) for value in values}
        cluster.discard("")
        if len(cluster) < 2:
            raise LexiconError(f"morphology entry {key!r} twins a term with itself.")
        groups.append(cluster)
    known_forms = set(known)
    twins: dict[str, frozenset[str]] = {}
    for cluster in groups:
        for member in cluster:
            twins[member] = frozenset(twins.get(member, frozenset()) | (cluster - {member}))
    unreachable = [key for key in data if normalize(key) not in known_forms]
    if unreachable:
        raise LexiconError(
            "morphology keys are not surface forms of any concept: " + ", ".join(sorted(unreachable))
        )
    return twins


def load_lexicon(path: str | Path) -> Lexicon:
    """Read and validate one role lexicon."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise LexiconError(f"no lexicon at {source}") from exc
    except json.JSONDecodeError as exc:
        raise LexiconError(f"lexicon at {source} is not valid JSON: {exc}") from exc
    return build_lexicon(payload)


def build_lexicon(payload: Any) -> Lexicon:
    """Validate an already-parsed lexicon document."""
    data = _require_mapping(payload, "lexicon")
    role = str(data.get("role", "")).strip()
    if not role:
        raise LexiconError("lexicon declares no role.")
    raw_concepts = data.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise LexiconError(f"lexicon {role!r} declares no concepts.")

    seen_ids: set[str] = set()
    concepts = tuple(_load_concept(item, seen_ids) for item in raw_concepts)

    index: dict[str, tuple[Concept, Form]] = {}
    for concept in concepts:
        for form in concept.forms:
            key = form.normalized
            if key in index:
                raise LexiconError(
                    f"surface form {key!r} is claimed by two concepts: "
                    f"{index[key][0].id!r} and {concept.id!r}."
                )
            index[key] = (concept, form)

    morphology = _load_morphology(data.get("morphology"), index)
    verbs = _load_term_set(data, "generic_verbs")
    umbrella = _load_term_set(data, "umbrella_terms")
    overlap = verbs & umbrella
    if overlap:
        raise LexiconError(
            "a term is both a ubiquitous verb and an umbrella term: "
            + ", ".join(sorted(overlap))
        )
    raw_parents = data.get("family_parents") or {}
    if not isinstance(raw_parents, Mapping):
        raise LexiconError("family_parents must be an object.")
    declared_families = {family for concept in concepts for family in concept.families}
    parents = {str(key): str(value) for key, value in raw_parents.items()}
    unknown = sorted(set(parents) - declared_families)
    if unknown:
        raise LexiconError(
            "family_parents names families no concept declares: " + ", ".join(unknown)
        )

    return Lexicon(
        role=role,
        version=int(data.get("version", 1)),
        concepts=concepts,
        morphology=morphology,
        generic_verbs=verbs,
        umbrella_terms=umbrella,
        employer_terms=_load_term_set(data, "employer_terms"),
        noisy_terms=_load_term_set(data, "noisy_terms"),
        family_relevance=_load_family_relevance(data, declared_families),
        anchor_families=frozenset(
            str(family) for family in (data.get("anchor_families") or [])
        ),
        neutral_variant=float(data.get("neutral_variant", 0.35)),
        family_parents=parents,
        note=str(data.get("note", "")),
        _form_index=index,
    )


def _load_family_relevance(
    data: Mapping[str, Any], declared: set[str]
) -> dict[str, str]:
    raw = data.get("family_relevance") or {}
    if not isinstance(raw, Mapping):
        raise LexiconError("family_relevance must be an object.")
    mapping: dict[str, str] = {}
    for family, level in raw.items():
        name, value = str(family), str(level).strip()
        if value not in RELEVANCE_KEYS:
            raise LexiconError(
                f"family {name!r} declares an unknown relevance {value!r}."
            )
        if name not in declared:
            raise LexiconError(
                f"family_relevance names a family no concept declares: {name!r}"
            )
        mapping[name] = value
    return mapping


def _load_term_set(data: Mapping[str, Any], key: str) -> frozenset[str]:
    raw = data.get(key) or []
    if not isinstance(raw, list):
        raise LexiconError(f"{key} must be a list.")
    return frozenset(normalize(term) for term in raw if normalize(term))
