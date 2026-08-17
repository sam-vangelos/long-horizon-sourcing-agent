from __future__ import annotations

import pytest

from linkedin.boolean_compiler import (
    DEFAULT_MAX_VARIANTS_PER_GROUP,
    BooleanNormalizationError,
    UbiquitousAndGateError,
    derive_surface_variants,
    normalize_boolean_for_linkedin,
    normalize_execution_work_item_boolean,
)


def test_normalizer_byte_identical_default_for_plain_keyword_boolean() -> None:
    report = normalize_boolean_for_linkedin('("Python" OR "PyTorch") AND ("production")')

    assert report.normalized_boolean == report.original_boolean
    assert report.changed is False
    assert report.findings == ()


def test_normalizer_strips_terms_already_carried_by_structured_filters() -> None:
    report = normalize_boolean_for_linkedin(
        '("Nubank" OR "Bancolombia" OR "fintech") AND ("ML Engineer" OR "platform")',
        structured_filters={
            "companies": ["Nubank", "Bancolombia"],
            "titles": ["ML Engineer"],
        },
    )

    assert report.normalized_boolean == '("fintech") AND ("platform")'
    assert [finding.code for finding in report.findings] == [
        "surface_conflict_stripped",
        "surface_conflict_stripped",
        "surface_conflict_stripped",
    ]


def test_normalizer_token_subset_pruning_is_explicitly_enabled() -> None:
    pending = normalize_boolean_for_linkedin(
        '("reward model" OR "reward model development")'
    )
    pruned = normalize_boolean_for_linkedin(
        '("reward model" OR "reward model development")',
        enable_token_subset_pruning=True,
    )

    assert pending.normalized_boolean == '("reward model" OR "reward model development")'
    assert pending.findings == ()
    assert pruned.normalized_boolean == '("reward model")'
    assert [finding.code for finding in pruned.findings] == [
        "token_subset_superstring_pruned"
    ]


def test_normalizer_flags_ubiquitous_and_gate_from_explicit_term_set() -> None:
    report = normalize_boolean_for_linkedin(
        '("Python") AND ("PyTorch")',
        ubiquitous_terms={"python", "pytorch"},
    )

    assert report.normalized_boolean == '("Python") AND ("PyTorch")'
    assert [finding.code for finding in report.findings] == ["ubiquitous_and_gate"]


def test_normalizer_rejects_structured_filter_values_that_would_be_stringified() -> None:
    with pytest.raises(BooleanNormalizationError, match="structured_filters.companies"):
        normalize_boolean_for_linkedin(
            '("123")',
            structured_filters={"companies": [123]},
        )


def test_execution_normalizer_rejects_malformed_ubiquitous_terms() -> None:
    item = {
        "boolean": '("Python") AND ("PyTorch")',
        "ubiquitous_terms": ["python", 123],
    }

    with pytest.raises(BooleanNormalizationError, match="ubiquitous_terms"):
        normalize_execution_work_item_boolean(item, boolean_key="boolean")


def test_execution_normalizer_unions_explicit_and_item_supplied_ubiquitous_terms() -> None:
    # Neither source alone covers both AND-clause terms, so the gate must not
    # fire from either in isolation...
    item_partial = {
        "boolean": '("Python") AND ("PyTorch")',
        "ubiquitous_terms": ["pytorch"],
    }
    result = normalize_execution_work_item_boolean(
        item_partial,
        boolean_key="boolean",
        ubiquitous_terms={"engineer"},
    )
    assert result["boolean"] == '("Python") AND ("PyTorch")'

    # ...but the UNION of the explicit brief-derived feed and the item-supplied
    # term does cover both clauses, so the execution seam must raise.
    item_union = {
        "boolean": '("Python") AND ("PyTorch")',
        "ubiquitous_terms": ["pytorch"],
    }
    with pytest.raises(BooleanNormalizationError, match="ubiquitous-term AND-gate"):
        normalize_execution_work_item_boolean(
            item_union,
            boolean_key="boolean",
            ubiquitous_terms={"python"},
        )


# --- deterministic surface-variant expansion -------------------------------
# The strategy doctrine mandates hyphenation variants per OR group and forbids
# fabricated variants for proper-noun tools. Two frontier models at max effort
# delivered 0 of 18 and 11 of 22 hyphenated terms with their spaced twin, so
# that form is derived mechanically instead of being asked for.
#
# Expansion covers ONE axis in ONE direction — dehyphenation only
# (2026-07-30 audit: reverse synthesis invented "san-francisco" and
# "new-york" from arbitrary spaced phrases) — because that transform is
# orthographic. The number axis was built and removed the same day: pluralising
# is morphological and needs a lexicon, and rules alone turned real tooling
# vocabulary into "kubernete", "mlop", "verls" and "datas". The tests below lock
# the removal, not just the remaining behaviour — a reintroduced pluraliser
# fails several of them.


def test_expansion_is_off_by_default_and_byte_identical() -> None:
    boolean = '("post-training" OR "coding agent")'
    report = normalize_boolean_for_linkedin(boolean)

    assert report.normalized_boolean == boolean
    assert report.changed is False


def test_expansion_adds_the_dehyphenated_form_fable_omitted() -> None:
    report = normalize_boolean_for_linkedin(
        '("post-training" OR "fine-tuning")',
        expand_surface_variants=True,
    )

    assert '"post training"' in report.normalized_boolean
    assert '"fine tuning"' in report.normalized_boolean
    codes = [f.code for f in report.findings]
    assert "morphological_variants_added" in codes


def test_expansion_never_varies_a_proper_noun_tool() -> None:
    report = normalize_boolean_for_linkedin(
        '("SWE-Gym" OR "veRL")',
        expand_surface_variants=True,
        proper_nouns={"SWE-Gym", "veRL"},
    )

    assert report.normalized_boolean == '("SWE-Gym" OR "veRL")'
    assert report.findings == ()


def test_expansion_refuses_terms_carrying_digits_or_operators() -> None:
    # Second line of defence for artifacts an unpopulated brief never listed.
    report = normalize_boolean_for_linkedin(
        '("pass@k" OR "gpt-4" OR "c++")',
        expand_surface_variants=True,
    )

    assert report.normalized_boolean == '("pass@k" OR "gpt-4" OR "c++")'


@pytest.mark.parametrize(
    "term",
    ["kubernetes", "mlops", "devops", "veRL", "pytorch", "numpy", "jax", "data"],
)
def test_expansion_never_guesses_a_number_form(term: str) -> None:
    # These are the exact strings a rules-only pluraliser mangled on 2026-07-27
    # ("kubernete", "mlop", "devop", "verls", "pytorches", "numpies", "jaxes",
    # "datas"). None carries a digit or an operator, so the punctuation guard
    # does not save them and the brief's do-not-vary field is empty in
    # production — removing the axis is the only thing that does.
    assert derive_surface_variants(term) == ()

    report = normalize_boolean_for_linkedin(
        f'("{term}")', expand_surface_variants=True
    )
    assert report.normalized_boolean == f'("{term}")'
    assert report.findings == ()


def test_expansion_does_not_fabricate_a_compound_gerund_plural() -> None:
    report = normalize_boolean_for_linkedin(
        '("post-training" OR "reward modeling")',
        expand_surface_variants=True,
    )

    assert "post trainings" not in report.normalized_boolean
    assert "reward modelings" not in report.normalized_boolean
    # The spaced twin IS still owed — removing the number axis must not take
    # the hyphenation axis with it.
    assert '"post training"' in report.normalized_boolean


def test_expansion_does_not_disarm_the_ubiquitous_and_gate() -> None:
    # Regression: expansion inserted "engineers"/"technologies", neither of
    # which is in the ubiquity feed, so every group stopped being "composed
    # entirely of ubiquitous terms" and a refusal became a pass. A derived
    # surface form of a ubiquitous term carries no new meaning and must not
    # count as fresh vocabulary.
    boolean = '("engineer") AND ("post-training")'
    ubiquitous = {"engineer", "post-training"}

    baseline = normalize_boolean_for_linkedin(boolean, ubiquitous_terms=ubiquitous)
    assert "ubiquitous_and_gate" in [f.code for f in baseline.findings]

    expanded = normalize_boolean_for_linkedin(
        boolean, ubiquitous_terms=ubiquitous, expand_surface_variants=True
    )
    assert '"post training"' in expanded.normalized_boolean
    assert "ubiquitous_and_gate" in [f.code for f in expanded.findings]


def test_expansion_gate_still_raises_through_the_execution_seam() -> None:
    # The gate is only a refusal because this seam turns it into one.
    with pytest.raises(UbiquitousAndGateError):
        normalize_execution_work_item_boolean(
            {"boolean": '("engineer") AND ("post-training")'},
            boolean_key="boolean",
            ubiquitous_terms={"engineer", "post-training"},
            expand_surface_variants=True,
        )


def test_expansion_composed_with_pruning_keeps_the_authored_phrase() -> None:
    # Pruning is ON by default at the execution seam, and it was never tested
    # together with expansion. It must not become a route by which a widening
    # pass narrows the string: every term the model wrote survives, and the
    # derived form is added alongside.
    item = normalize_execution_work_item_boolean(
        {"boolean": '("post-training" OR "llm post training")'},
        boolean_key="boolean",
        expand_surface_variants=True,
    )

    assert '"post-training"' in item["boolean"]
    assert '"llm post training"' in item["boolean"]
    assert '"post training"' in item["boolean"]
    pruned = [
        f for f in item["boolean_normalization"]["findings"]
        if f["code"] == "token_subset_superstring_pruned"
    ]
    assert not pruned


@pytest.mark.parametrize(
    "boolean,authored,derived",
    [
        ('("post-training" OR "llm post training")', "llm post training", "post training"),
        ('("coding-agent" OR "autonomous coding agent")', "autonomous coding agent", "coding agent"),
        ('("fine-tuning" OR "supervised fine tuning")', "supervised fine tuning", "fine tuning"),
    ],
)
def test_a_derived_variant_never_prunes_an_authored_term(
    boolean: str, authored: str, derived: str
) -> None:
    # Regression, found by attacking the fix that preceded it. Dehyphenating
    # "post-training" yields "post training", whose TOKENS are a proper subset
    # of the authored "llm post training" — so the subset pruner deleted the
    # model's most specific phrase on the authority of a term the compiler had
    # just invented, and a widening pass shipped a narrower string. Authored
    # terms may prune; derived respellings may not.
    report = normalize_boolean_for_linkedin(
        boolean, enable_token_subset_pruning=True, expand_surface_variants=True
    )

    assert f'"{authored}"' in report.normalized_boolean
    assert f'"{derived}"' in report.normalized_boolean
    assert not [
        f for f in report.findings if f.code == "token_subset_superstring_pruned"
    ]


@pytest.mark.parametrize(
    "term,expected",
    [
        ("post - training", ("post training",)),   # not "post   training"
        ("-", ()),                                  # not (" ",)
        ("--", ()),
        ("-lead", ()),                              # edge hyphen: refused, not (" lead",)
        ("trail-", ()),                             # edge hyphen: refused, not ("trail ",)
        ("a-b-c", ("a b c",)),
        ("machine  learning", ()),  # reverse synthesis removed 2026-07-30: spaced phrases derive nothing
    ],
)
def test_derived_variant_is_always_whitespace_normalized(
    term: str, expected: tuple[str, ...]
) -> None:
    # Swapping a hyphen for a space can leave whitespace the input never had.
    # Every one of these was quoted verbatim into the output boolean before the
    # derived form was re-normalized, producing terms that match nobody.
    assert derive_surface_variants(term) == expected
    for variant in derive_surface_variants(term):
        assert variant == " ".join(variant.split())
        assert variant.strip() == variant


# Captured from the compiler as it stood BEFORE surface expansion existed
# (git show HEAD:linkedin/boolean_compiler.py, HEAD = ce5ffa8). This is the
# oracle: comparing the current function against itself with the flag omitted
# versus explicitly False proves nothing, because both take the same branch.
_PRE_EXPANSION_GOLDEN: tuple[tuple[str, bool, str, tuple[str, ...]], ...] = (
    ('("Python" OR "PyTorch") AND ("production")', False,
     '("Python" OR "PyTorch") AND ("production")', ()),
    ('("Python" OR "PyTorch") AND ("production")', True,
     '("Python" OR "PyTorch") AND ("production")', ()),
    ('("agent" OR "coding agent")', False, '("agent" OR "coding agent")', ()),
    ('("agent" OR "coding agent")', True, '("agent")',
     ("token_subset_superstring_pruned",)),
    ('("post-training" OR "llm post training")', False,
     '("post-training" OR "llm post training")', ()),
    ('("post-training" OR "llm post training")', True,
     '("post-training" OR "llm post training")', ()),
    ('("LLM" OR "LLM training" OR "llm")', False, '("LLM" OR "LLM training")', ()),
    ('("LLM" OR "LLM training" OR "llm")', True, '("LLM")',
     ("token_subset_superstring_pruned",)),
    ('("SWE-bench" OR "SWE bench") AND NOT ("recruiter" OR "sales")', False,
     '("SWE-bench" OR "SWE bench") AND NOT ("recruiter" OR "sales")', ()),
    ('("SWE-bench" OR "SWE bench") AND NOT ("recruiter" OR "sales")', True,
     '("SWE-bench" OR "SWE bench") AND NOT ("recruiter" OR "sales")', ()),
    ('("engineer") AND ("technology")', False, '("engineer") AND ("technology")',
     ("ubiquitous_and_gate",)),
    ('("engineer") AND ("technology")', True, '("engineer") AND ("technology")',
     ("ubiquitous_and_gate",)),
    ('("pass@k" OR "gpt-4" OR "c++")', False, '("pass@k" OR "gpt-4" OR "c++")', ()),
    ('("pass@k" OR "gpt-4" OR "c++")', True, '("pass@k" OR "gpt-4" OR "c++")', ()),
    ('("a" OR "a b" OR "a b c" OR "b c")', False,
     '("a" OR "a b" OR "a b c" OR "b c")', ()),
    ('("a" OR "a b" OR "a b c" OR "b c")', True, '("a" OR "b c")',
     ("token_subset_superstring_pruned",)),
    ('("research-and-development")', False, '("research-and-development")', ()),
    ('("research-and-development")', True, '("research-and-development")', ()),
)


@pytest.mark.parametrize("boolean,pruning,expected,codes", _PRE_EXPANSION_GOLDEN)
def test_flag_off_matches_the_pre_expansion_compiler(
    boolean: str, pruning: bool, expected: str, codes: tuple[str, ...]
) -> None:
    # The expansion work changed _normalize_group_terms' return arity, added a
    # gate-group filter, moved the AND detection behind quote masking, and added
    # a `pruners` argument to the subset pruner. Production runs with the flag
    # OFF, so every one of those has to be inert there.
    report = normalize_boolean_for_linkedin(
        boolean,
        enable_token_subset_pruning=pruning,
        ubiquitous_terms={"engineer", "technology"},
    )

    assert report.normalized_boolean == expected
    assert tuple(sorted(f.code for f in report.findings)) == codes


def test_flag_off_ignores_an_item_supplied_proper_nouns_key() -> None:
    # The pre-change compiler had no `proper_nouns` concept, so a work item
    # carrying that key — malformed or not — was simply an unknown key. Reading
    # it unconditionally turned it into a raise on the production path.
    item = normalize_execution_work_item_boolean(
        {"boolean": '("coding agent")', "proper_nouns": 7},
        boolean_key="boolean",
    )

    assert item["boolean"] == '("coding agent")'


def test_a_quoted_conjunction_is_not_an_and_clause() -> None:
    # The gate tested `" AND " in text.upper()` against the RAW string, so a
    # term that merely contains the word read as an AND clause. Dehyphenating
    # "research-and-development" introduced exactly that, and the execution seam
    # turned the resulting finding into a raise on a single-group string.
    report = normalize_boolean_for_linkedin(
        '("research-and-development")',
        ubiquitous_terms={"research-and-development"},
        expand_surface_variants=True,
    )

    assert '"research and development"' in report.normalized_boolean
    assert "ubiquitous_and_gate" not in [f.code for f in report.findings]

    normalize_execution_work_item_boolean(
        {"boolean": '("research-and-development")'},
        boolean_key="boolean",
        ubiquitous_terms={"research-and-development"},
        expand_surface_variants=True,
    )


@pytest.mark.parametrize("term", ["A-", "-A", 'a"b', '"foo bar"'])
def test_expansion_refuses_edge_hyphens_and_embedded_quotes(term: str) -> None:
    # Removing an edge hyphen rewrites rather than respells ("A-" is a grade,
    # "a" is a different term), and a quote would be re-quoted by
    # _render_or_group into a malformed group.
    assert derive_surface_variants(term) == ()


def test_expansion_does_not_widen_a_not_clause_into_new_meaning() -> None:
    # Expansion runs inside NOT groups too, where a wrong term is a false
    # NEGATIVE — a wrongly excluded candidate is never seen again. The number
    # axis turned "sales" into "sale", which knocks out pre-sales engineers.
    # Hyphenation cannot do that: it only respells what is already excluded.
    report = normalize_boolean_for_linkedin(
        '("coding agent") AND NOT ("recruiter" OR "staffing" OR "sales")',
        expand_surface_variants=True,
    )

    assert '"sale"' not in report.normalized_boolean
    assert '"recruiters"' not in report.normalized_boolean
    assert '"staffings"' not in report.normalized_boolean


def test_expansion_is_bounded_per_group() -> None:
    # 15 two-word terms yield 15 candidate variants against a cap of 12.
    words = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar"
    )
    terms = " OR ".join(f'"{w}-widget"' for w in words.split())
    report = normalize_boolean_for_linkedin(
        f"({terms})", expand_surface_variants=True
    )
    added = [f for f in report.findings if f.code == "morphological_variants_added"]

    assert added, "expected an expansion finding"
    assert len(added[0].terms) == DEFAULT_MAX_VARIANTS_PER_GROUP


def test_expansion_receipts_every_added_term_on_the_work_item() -> None:
    item = normalize_execution_work_item_boolean(
        {"boolean": '("post-training")'},
        boolean_key="boolean",
        expand_surface_variants=True,
    )
    findings = item["boolean_normalization"]["findings"]
    added = [f for f in findings if f["code"] == "morphological_variants_added"]

    assert added and added[0]["terms"] == ["post training"]
    assert '"post training"' in item["boolean"]


def test_expansion_exempts_every_surface_form_of_a_named_artifact() -> None:
    # Regression: the live 2026-07-27 string carried both "SWE-bench" and
    # "SWE bench". Exempting only the declared hyphenated form left the spaced
    # form open to variation, which for a proper noun the doctrine forbids.
    report = normalize_boolean_for_linkedin(
        '("SWE-bench" OR "SWE bench")',
        expand_surface_variants=True,
        proper_nouns={"SWE-bench"},
    )

    assert report.normalized_boolean == '("SWE-bench" OR "SWE bench")'
    assert report.findings == ()


@pytest.mark.parametrize(
    "phrase", ["San Francisco", "New York", "machine learning", "reward modeling"]
)
def test_reverse_hyphen_synthesis_is_not_performed(phrase: str) -> None:
    # 2026-07-30 audit: dehyphenation and hyphenation are NOT symmetric.
    # Dehyphenating an authored compound yields a form people demonstrably
    # write; hyphenating an arbitrary spaced phrase invents one — "San
    # Francisco" became "san-francisco", pure noise in an OR group.
    assert derive_surface_variants(phrase) == ()


def test_dehyphenation_still_works_in_the_kept_direction() -> None:
    assert derive_surface_variants("post-training") == ("post training",)
    assert derive_surface_variants("fine-tuning") == ("fine tuning",)


@pytest.mark.parametrize(
    "phrase", ["San Francisco", "New York", "machine learning", "reward modeling"]
)
def test_reverse_hyphen_synthesis_is_not_performed(phrase: str) -> None:
    # 2026-07-30 audit: dehyphenation and hyphenation are NOT symmetric.
    # Dehyphenating an authored compound yields a form people demonstrably
    # write; hyphenating an arbitrary spaced phrase invents one — "San
    # Francisco" became "san-francisco", pure noise in an OR group.
    assert derive_surface_variants(phrase) == ()


def test_dehyphenation_still_works_in_the_kept_direction() -> None:
    assert derive_surface_variants("post-training") == ("post training",)
    assert derive_surface_variants("fine-tuning") == ("fine tuning",)
