"""Lexicon schema, loader validation, and lookup (plans/boolean-scorer.md §5).

Schema-shape assertions over the shipped role lexicon are marked ``structural``
per pyproject.toml: they assert on declared data, not on behaviour.
"""

from __future__ import annotations

import json

import pytest

from linkedin.boolean_score.lexicon import (
    CONCEPT_KIND_KEYS,
    RELEVANCE_KEYS,
    KIND_KEYS,
    LEXICON_DIR,
    REGISTER_KEYS,
    Lexicon,
    LexiconError,
    derived_hyphen_twin,
    load_lexicon,
)

ROLE = "research-engineer-post-training"
LEXICON_PATH = LEXICON_DIR / f"{ROLE}.json"


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return load_lexicon(LEXICON_PATH)


def _write(tmp_path, payload) -> str:
    path = tmp_path / "role.json"
    path.write_text(json.dumps(payload))
    return path


def _minimal(**overrides) -> dict:
    payload = {
        "role": "a-role",
        "version": 1,
        "concepts": [
            {
                "id": "alpha",
                "family": "a_family",
                "kind": "class",
                "forms": [{"text": "alpha", "register": "common", "kind": "noun"}],
            }
        ],
        "morphology": {},
        "generic_terms": [],
    }
    payload.update(overrides)
    return payload


# --- the shipped lexicon: declared shape ------------------------------------


@pytest.mark.structural
def test_shipped_lexicon_names_its_role(lexicon):
    assert lexicon.role == ROLE
    assert lexicon.version >= 1
    assert len(lexicon.concepts) >= 40


@pytest.mark.structural
def test_every_form_declares_a_known_register_and_kind(lexicon):
    for concept in lexicon.concepts:
        assert concept.forms, concept.id
        for form in concept.forms:
            assert form.register in REGISTER_KEYS, (concept.id, form.text)
            assert form.kind in KIND_KEYS, (concept.id, form.text)


@pytest.mark.structural
def test_every_concept_can_be_earned(lexicon):
    """A concept may carry no common form when the ruling is that nobody
    writes its vocabulary, but it must be earnable by something."""
    for concept in lexicon.concepts:
        assert concept.common_forms or concept.occasional_forms, concept.id


@pytest.mark.structural
def test_every_family_declares_its_relevance_to_the_role(lexicon):
    for concept in lexicon.concepts:
        assert lexicon.relevance_of(concept) in RELEVANCE_KEYS, concept.id
    assert lexicon.relevance_of(lexicon.concept_for("post-training")) == "core"
    assert lexicon.relevance_of(lexicon.concept_for("vector database")) == "peripheral"
    assert lexicon.relevance_of(lexicon.concept_for("VLM")) == "peripheral"
    assert lexicon.instances_in_family("benchmarks") >= 20


@pytest.mark.structural
def test_every_concept_declares_at_least_one_family(lexicon):
    for concept in lexicon.concepts:
        assert concept.families, concept.id
        assert isinstance(concept.families, tuple)
        for family in concept.families:
            assert family == family.strip().lower()
    assert len(lexicon.family_names) >= 8
    assert lexicon.families_for("post-training") == lexicon.families_for(
        "instruction tuning"
    )
    assert not set(lexicon.families_for("post-training")) & set(
        lexicon.families_for("research engineer")
    )


@pytest.mark.structural
def test_a_concept_may_declare_several_families(lexicon):
    assert len(lexicon.families_for("GRPO")) == 2
    assert set(lexicon.families_for("GRPO")) >= set(lexicon.families_for("RLHF"))


def test_family_accepts_a_string_or_a_list(tmp_path):
    payload = _minimal(
        concepts=[
            {
                "id": "alpha",
                "families": ["one", "two"],
                "kind": "class",
                "forms": [{"text": "alpha", "register": "common", "kind": "noun"}],
            },
            {
                "id": "beta",
                "family": "three",
                "kind": "instance",
                "forms": [{"text": "beta", "register": "common", "kind": "noun"}],
            },
        ]
    )
    loaded = load_lexicon(_write(tmp_path, payload))
    assert loaded.families_for("alpha") == ("one", "two")
    assert loaded.families_for("beta") == ("three",)


@pytest.mark.structural
def test_no_surface_form_is_shared_between_concepts(lexicon):
    seen: dict[str, str] = {}
    for concept in lexicon.concepts:
        for form in concept.forms:
            assert form.normalized not in seen, (form.text, concept.id, seen.get(form.normalized))
            seen[form.normalized] = concept.id


@pytest.mark.structural
def test_every_morphology_cluster_can_fire(lexicon):
    """A twin pair only does work when at least one side is a mapped form."""
    for key, twins in lexicon.morphology.items():
        assert twins, key
        cluster = {key} | set(twins)
        assert any(lexicon.concept_for(member) for member in cluster), key


@pytest.mark.structural
def test_ubiquitous_vocabulary_is_declared_in_two_registers(lexicon):
    assert len(lexicon.generic_verbs) >= 6
    assert len(lexicon.umbrella_terms) >= 20
    for term in lexicon.generic_verbs | lexicon.umbrella_terms:
        assert term == term.lower().strip()
    assert not lexicon.generic_verbs & lexicon.umbrella_terms


@pytest.mark.structural
def test_every_concept_declares_whether_it_is_a_class_or_an_instance(lexicon):
    for concept in lexicon.concepts:
        assert concept.kind in CONCEPT_KIND_KEYS, concept.id
        if concept.proper_noun:
            assert concept.kind == "instance", concept.id
    kinds = {concept.kind for concept in lexicon.concepts}
    assert kinds == CONCEPT_KIND_KEYS


@pytest.mark.structural
def test_family_parents_gather_the_post_training_practice(lexicon):
    assert lexicon.family_parents
    parents = set(lexicon.family_parents.values())
    assert len(parents) >= 1
    assert lexicon.parent_of("evaluation") == lexicon.parent_of("data_curation")
    assert lexicon.parent_of("titles") is None


def test_a_proper_noun_declared_as_a_class_is_rejected(tmp_path):
    payload = _minimal(
        concepts=[
            {
                "id": "alpha",
                "family": "f",
                "kind": "class",
                "proper_noun": True,
                "forms": [{"text": "alpha", "register": "common", "kind": "noun"}],
            }
        ]
    )
    with pytest.raises(LexiconError, match="instance"):
        load_lexicon(_write(tmp_path, payload))


def test_a_concept_without_a_kind_is_rejected(tmp_path):
    payload = _minimal(
        concepts=[
            {
                "id": "alpha",
                "family": "f",
                "forms": [{"text": "alpha", "register": "common", "kind": "noun"}],
            }
        ]
    )
    with pytest.raises(LexiconError, match="class"):
        load_lexicon(_write(tmp_path, payload))


@pytest.mark.structural
def test_unsure_items_are_listed_for_review(lexicon):
    unsure = lexicon.unsure_items()
    assert unsure, "the first draft must flag what Sam has to review"
    assert all(isinstance(item, str) for item in unsure)


# --- loader validation ------------------------------------------------------


def test_duplicate_form_across_concepts_is_rejected(tmp_path):
    payload = _minimal(
        concepts=[
            {"id": "alpha", "family": "f", "kind": "class", "forms": [{"text": "shared", "register": "common", "kind": "noun"}]},
            {"id": "beta", "family": "f", "kind": "class", "forms": [{"text": "Shared", "register": "common", "kind": "noun"}]},
        ]
    )
    with pytest.raises(LexiconError, match="shared"):
        load_lexicon(_write(tmp_path, payload))


def test_duplicate_concept_id_is_rejected(tmp_path):
    payload = _minimal(
        concepts=[
            {"id": "alpha", "family": "f", "kind": "class", "forms": [{"text": "one", "register": "common", "kind": "noun"}]},
            {"id": "alpha", "family": "f", "kind": "class", "forms": [{"text": "two", "register": "common", "kind": "noun"}]},
        ]
    )
    with pytest.raises(LexiconError, match="alpha"):
        load_lexicon(_write(tmp_path, payload))


def test_unknown_register_is_rejected(tmp_path):
    payload = _minimal(
        concepts=[{"id": "alpha", "family": "f", "kind": "class", "forms": [{"text": "one", "register": "frequent", "kind": "noun"}]}]
    )
    with pytest.raises(LexiconError, match="register"):
        load_lexicon(_write(tmp_path, payload))


def test_unknown_kind_is_rejected(tmp_path):
    payload = _minimal(
        concepts=[{"id": "alpha", "family": "f", "kind": "class", "forms": [{"text": "one", "register": "common", "kind": "verb"}]}]
    )
    with pytest.raises(LexiconError, match="kind"):
        load_lexicon(_write(tmp_path, payload))


def test_concept_without_a_family_is_rejected(tmp_path):
    payload = _minimal(
        concepts=[{"id": "alpha", "forms": [{"text": "one", "register": "common", "kind": "noun"}]}]
    )
    with pytest.raises(LexiconError, match="family"):
        load_lexicon(_write(tmp_path, payload))


def test_missing_role_is_rejected(tmp_path):
    payload = _minimal()
    del payload["role"]
    with pytest.raises(LexiconError, match="role"):
        load_lexicon(_write(tmp_path, payload))


def test_concept_without_forms_is_rejected(tmp_path):
    payload = _minimal(concepts=[{"id": "alpha", "family": "f", "kind": "class", "forms": []}])
    with pytest.raises(LexiconError, match="alpha"):
        load_lexicon(_write(tmp_path, payload))


# --- lookup -----------------------------------------------------------------


def test_lookup_normalizes_case_and_whitespace(lexicon):
    direct = lexicon.concept_for("post-training")
    assert direct is not None
    assert lexicon.concept_for("  POST-TRAINING ") is direct
    spaced = lexicon.concept_for("large   language\tmodel")
    assert spaced is not None and spaced.id == lexicon.concept_for("large language model").id


def test_unknown_term_maps_to_nothing(lexicon):
    assert lexicon.concept_for("Acme Robotics International") is None
    assert lexicon.form_for("Acme Robotics International") is None


def test_expected_forms_are_the_common_ones(lexicon):
    concept = lexicon.concept_for("evaluation")
    expected = {form.normalized for form in lexicon.expected_forms(concept)}
    assert expected == {"evaluation", "evaluations", "eval", "evals"}
    assert all(form.register == "common" for form in lexicon.expected_forms(concept))


def test_registers_partition_a_concept(lexicon):
    concept = lexicon.concept_for("RLHF")
    assert {form.register for form in concept.forms} <= REGISTER_KEYS
    assert len(concept.forms) == len(concept.common_forms) + len(concept.occasional_forms) + len(
        concept.rare_forms
    )


def test_ubiquitous_term_lookup(lexicon):
    assert lexicon.is_umbrella("Engineer") is True
    assert lexicon.is_umbrella("Machine Learning") is True
    assert lexicon.is_umbrella("reward model") is False
    assert lexicon.is_generic_verb("Led") is True
    assert lexicon.is_generic_verb("engineer") is False


def test_proper_noun_flag_round_trips(lexicon):
    assert lexicon.concept_for("SWE-bench").proper_noun is True
    assert lexicon.concept_for("reward model").proper_noun is False


# --- morphology and derived twins -------------------------------------------


@pytest.mark.parametrize(
    ("term", "twin"),
    [
        ("post-training", "post training"),
        ("LLM-as-a-judge", "llm as a judge"),
        ("open-weights", "open weights"),
    ],
)
def test_derived_hyphen_twin(term, twin):
    assert derived_hyphen_twin(term) == twin


@pytest.mark.parametrize(
    "term",
    ["GPT-4", "Qwen2.5-Coder", "MiniWoB++", "c#-shop", "user@host-name", "post training", "alpha-", "-alpha"],
)
def test_no_derived_twin(term):
    assert derived_hyphen_twin(term) is None


def test_morphology_is_symmetric(lexicon):
    assert "benchmarks" in lexicon.twins_for("benchmark")
    assert "benchmark" in lexicon.twins_for("benchmarks")


def test_twins_combine_morphology_and_the_hyphen_axis(lexicon):
    twins = lexicon.twins_for("post-training")
    assert "post training" in twins


def test_term_without_twins_has_none(lexicon):
    assert lexicon.twins_for("RLHF") == frozenset()
