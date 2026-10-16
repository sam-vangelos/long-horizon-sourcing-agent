"""The deterministic enricher: what it may add, what it must never touch.

The enricher is the inverse of `tools/make_boolean_pairs.py`'s degradation
operators. Every rule below is a rule about what it is NOT allowed to do,
because the thing it is wired into builds live production plans.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linkedin.boolean_score.enrich import enrich
from linkedin.boolean_score.lexicon import (
    LEXICON_DIR,
    RARE_REGISTER,
    load_lexicon,
    normalize,
)
from linkedin.boolean_score.parser import parse
from linkedin.boolean_score.score import score

CORPUS = (
    Path(__file__).resolve().parent / "fixtures" / "boolean_corpus" / "ml_runs_76.json"
)


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon(LEXICON_DIR / "research-engineer-post-training.json")


@pytest.fixture(scope="module")
def corpus():
    return json.loads(CORPUS.read_text())["strings"]


def _surfaces(boolean: str) -> set[str]:
    result = parse(boolean)
    if not result.legal:
        return set()
    from linkedin.boolean_score.score import _groups_for

    return {
        term.normalized
        for group in _groups_for(result.tree, _LEX[0])
        for term in group.terms
    }


_LEX: list = []


@pytest.fixture(autouse=True)
def _bind_lexicon(lexicon):
    _LEX.clear()
    _LEX.append(lexicon)


# --- the contract -----------------------------------------------------------


def test_an_illegal_string_comes_back_untouched(lexicon):
    report = enrich('("unclosed OR "x")', lexicon)
    assert report.legal is False
    assert report.enriched == '("unclosed OR "x")'
    assert report.unchanged is True
    assert report.changes == ()


def test_enrichment_is_idempotent_over_the_corpus(lexicon, corpus):
    for entry in corpus:
        once = enrich(entry["boolean"], lexicon).enriched
        twice = enrich(once, lexicon).enriched
        assert twice == once, f"not idempotent: {entry['id']}"


def test_enrichment_never_makes_a_legal_string_illegal(lexicon, corpus):
    for entry in corpus:
        original = entry["boolean"]
        if not parse(original).legal:
            continue
        enriched = enrich(original, lexicon).enriched
        result = parse(enriched)
        assert result.legal, f"{entry['id']} became illegal: {enriched}"


def test_enrichment_never_removes_a_non_rare_authored_term(lexicon, corpus):
    for entry in corpus:
        original = entry["boolean"]
        if not parse(original).legal:
            continue
        before = _surfaces(original)
        after = _surfaces(enrich(original, lexicon).enriched)
        for surface in before - after:
            form = lexicon.form_for(surface)
            assert form is not None and form.register == RARE_REGISTER, (
                f"{entry['id']} dropped non-rare term {surface!r}"
            )


def test_enrichment_never_touches_a_not_group(lexicon):
    boolean = '("post-training" OR "RLHF") NOT ("recruiter" OR "sales")'
    report = enrich(boolean, lexicon)
    tail = report.enriched.split("NOT", 1)[1]
    assert tail == boolean.split("NOT", 1)[1]


def test_enrichment_respects_the_character_budget(lexicon):
    boolean = '("post-training" OR "RLHF")'
    report = enrich(boolean, lexicon, max_chars=len(boolean))
    assert len(report.enriched) <= len(boolean)
    assert report.truncated is True


def test_enrichment_never_adds_an_umbrella_or_noisy_term(lexicon, corpus):
    for entry in corpus:
        original = entry["boolean"]
        if not parse(original).legal:
            continue
        added = _surfaces(enrich(original, lexicon).enriched) - _surfaces(original)
        for surface in added:
            assert not lexicon.is_umbrella(surface), f"added umbrella {surface!r}"
            assert not lexicon.is_noisy(surface), f"added noisy {surface!r}"
            assert not lexicon.is_generic_verb(surface), f"added verb {surface!r}"
            form = lexicon.form_for(surface)
            assert form is None or form.register != RARE_REGISTER, (
                f"added rare form {surface!r}"
            )


def test_enrichment_never_changes_group_structure(lexicon, corpus):
    from linkedin.boolean_score.score import _groups_for

    for entry in corpus:
        original = entry["boolean"]
        before = parse(original)
        if not before.legal:
            continue
        after = parse(enrich(original, lexicon).enriched)
        old = _groups_for(before.tree, lexicon)
        new = _groups_for(after.tree, lexicon)
        assert len(old) == len(new), f"{entry['id']} changed group count"
        for a, b in zip(old, new):
            assert a.negated == b.negated
            assert a.surfaces <= b.surfaces or all(
                lexicon.form_for(s) is not None
                and lexicon.form_for(s).register == RARE_REGISTER
                for s in a.surfaces - b.surfaces
            )


# --- the measurement claim --------------------------------------------------


def test_enrichment_raises_the_composite_on_at_least_95_percent(lexicon, corpus):
    """Every exception is a finding for Stage 3, not a number to tune."""
    import math

    exceptions = []
    scored = 0
    for entry in corpus:
        original = entry["boolean"]
        if not parse(original).legal:
            continue
        scored += 1
        before = score(original, lexicon)
        after = score(enrich(original, lexicon).enriched, lexicon)
        if after.log_composite < before.log_composite - 1e-9:
            exceptions.append(
                {
                    "id": entry["id"],
                    "before": round(before.composite, 4),
                    "after": round(after.composite, 4),
                    "bloat_before": {
                        k: round(v, 4)
                        for k, v in before.bloat.items()
                        if v
                    },
                    "bloat_after": {
                        k: round(v, 4) for k, v in after.bloat.items() if v
                    },
                }
            )
    ratio = 1 - (len(exceptions) / scored)
    assert ratio >= 0.95, (
        f"only {ratio:.1%} of {scored} strings improved or held; "
        f"exceptions:\n{json.dumps(exceptions, indent=2)}"
    )


def test_enrichment_reproduces_the_pair_operators(lexicon):
    """`enrich` is the inverse of the degradation operators in
    tools/make_boolean_pairs.py: applied to a degraded base it restores it."""
    from tools import make_boolean_pairs as pairs

    base = '("post-training" OR "RLHF" OR "DPO")'
    for name in ("add_missing_common_forms", "add_hyphen_twins"):
        operator = getattr(pairs, name, None)
        if operator is None:
            pytest.skip(f"{name} is not a module-level operator")
        produced = operator(base, lexicon)
        if not produced:
            continue
        enriched = enrich(base, lexicon).enriched
        assert _surfaces(produced) <= _surfaces(enriched), (
            f"{name} produced surfaces enrich() does not"
        )
