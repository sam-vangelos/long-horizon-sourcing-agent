"""The facet typeahead match rule, its name normalization, and its cache.

Sam's ruling 2026-09-20: TAKE THE NEAR MISS. LinkedIn Recruiter's typeahead
answers a strategist's "Mercado Livre" with "Mercado Livre Brasil", "Mercado
Libre", "RBE | Mercado Livre de Energia"; the exact-only rule threw all three
away and roughly three requested names in ten never landed. These tests pin
the rule that decides which offered option a request takes, the substitution
record that keeps the receipt honest about what landed, and the sidecar cache
that spends LinkedIn's own spelling from the second string on.
"""

from __future__ import annotations

import json

import pytest

from linkedin.browser import (
    facet_name_key,
    facet_name_tokens,
    facet_target_landed,
    format_facet_substituted_console_line,
    read_facet_name_resolutions,
    resolve_facet_option,
    write_facet_name_resolution,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Mercado Livre", ("mercado", "livre")),
        ("  Mercado   Livre  ", ("mercado", "livre")),
        ("Itaú", ("itau",)),
        ("Itau", ("itau",)),
        ("RBE | Mercado Livre", ("rbe", "mercado", "livre")),
        ("Principal Architect, IT", ("principal", "architect", "it")),
        ("", ()),
        (None, ()),
    ],
)
def test_facet_name_tokens_fold_case_accents_and_punctuation(value, expected):
    assert facet_name_tokens(value) == expected


def test_facet_name_key_is_the_token_sequence_joined():
    assert facet_name_key("  Mercado   LIVRE ") == "mercado livre"


def test_diacritics_do_not_separate_two_spellings_of_one_company():
    assert facet_name_key("Itaú Unibanco") == facet_name_key("Itau Unibanco")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("C++ Engineer", ("c++", "engineer")),
        ("Staff+", ("staff+",)),
        (".NET Developer", (".net", "developer")),
        ("C# Engineer", ("c#", "engineer")),
        ("&Open", ("&open",)),
        # Still trimmed: punctuation that dresses a token rather than naming it.
        ("Principal Architect, IT", ("principal", "architect", "it")),
        ("Banco do Brasil S.A.", ("banco", "do", "brasil", "s.a")),
    ],
)
def test_punctuation_that_carries_the_name_survives_the_fold(value, expected):
    """`+`, a leading `.` and a leading `#` are part of the entity's name.

    The edge-strip set took `+` and `&` from both edges, so "C++ Engineer"
    folded to "c engineer" and "&Open" to "open"; `.` is still trimmed from a
    token's trailing edge but no longer from its leading one, so ".NET
    Developer" is not "net developer".
    """
    assert facet_name_tokens(value) == expected


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("C++ Engineer", "C Engineer"),
        ("Staff+ Engineer", "Staff Engineer"),
        (".NET Developer", "NET Developer"),
        ("&Open", "Open"),
    ],
)
def test_a_name_and_its_punctuation_stripped_twin_are_two_entities(one, other):
    """The fold decides three things at once, and it collapsed all of them.

    `facet_name_key` keys the resolution cache, the typeahead's exact scan and
    the reconcile's owned index. With `+` stripped, a desired set holding both
    "C++ Engineer" and "C Engineer" raised `titles_desired_normalized_collision`
    and the string never ran; two such pills on the rail raised the observed
    twin, which re-anchors and counts toward the three-strike stop.
    """
    assert facet_name_key(one) != facet_name_key(other)


# ---------------------------------------------------------------------------
# The match rule
# ---------------------------------------------------------------------------


def test_exact_option_wins_over_a_containing_option_listed_first():
    """Rule 1 scans EVERY option before rule 2 takes the first container.

    LinkedIn ranks "Senior Software Engineer" above "Software Engineer" often
    enough that a first-match-wins reading of the rule would substitute a
    title the strategist did not ask for while the exact one sat one row down.
    """
    resolved = resolve_facet_option(
        "Software Engineer",
        ["Senior Software Engineer", "Software Engineer"],
    )
    assert resolved == (1, "Software Engineer")


def test_first_containing_option_in_linkedin_order_wins():
    """Rule 2: LinkedIn's ranking decides between two containing options."""
    resolved = resolve_facet_option(
        "Mercado Livre",
        ["Mercado Livre Brasil", "Mercado Libre", "RBE | Mercado Livre de Energia"],
    )
    assert resolved == (0, "Mercado Livre Brasil")


def test_containment_must_be_contiguous_and_in_order():
    """"Architect Principal" does not contain "Principal Architect"."""
    assert (
        resolve_facet_option("Principal Architect", ["Architect Principal"]) is None
    )


def test_a_trailing_qualifier_still_contains_the_request():
    resolved = resolve_facet_option(
        "Principal Architect", ["Principal Architect, IT"]
    )
    assert resolved == (0, "Principal Architect, IT")


def test_a_container_beats_a_prefix_whatever_linkedin_ranked_first():
    """Classes, not rows: a container keeps every requested token, a prefix
    throws one away, so the two are never ranked against each other.

    The live listbox from Addendum G5, both ways round. A row-ranked pass
    answered "Tess" when LinkedIn happened to list it first, which drops the
    "AI" the strategist asked for; the container keeps it, and the near-miss
    console line is the operator's tripwire if the sub-brand is wrong.
    """
    assert resolve_facet_option("Tess AI", ["Tess", "TESS AI Quality"]) == (
        1,
        "TESS AI Quality",
    )
    assert resolve_facet_option("Tess AI", ["TESS AI Quality", "Tess"]) == (
        0,
        "TESS AI Quality",
    )


def test_a_container_beats_a_prefix_that_is_a_different_company():
    """The row-ranked pass took "Mercado" — a company nobody asked for.

    "Mercado" is a four-character prefix of "Mercado Livre" and LinkedIn
    listed it first, so the shared pass returned it and never looked at the
    Brasil row. Under class ranking the container wins and the prefix is only
    reached when no option contains the request.
    """
    assert resolve_facet_option(
        "Mercado Livre", ["Mercado", "Mercado Livre Brasil"]
    ) == (1, "Mercado Livre Brasil")


def test_prefix_rule_refuses_a_prefix_under_four_characters():
    """"AI" must never become the whole company for "AI Lab"."""
    assert resolve_facet_option("AI Lab", ["AI"]) is None
    # The guard skips the two-character prefix and rule 2 takes the container.
    assert resolve_facet_option("AI Lab", ["AI", "AI Lab Systems"]) == (
        1,
        "AI Lab Systems",
    )


def test_a_request_under_four_characters_only_ever_takes_an_exact_row():
    """An initialism is not enough evidence to widen or narrow a search.

    "AI" contains nothing and prefixes nothing safely: every option that
    carries the token is a different entity. The guard is on the REQUEST, so
    it holds for containment as well as for the prefix rule — the row-length
    guard alone let "AI" take "AI Lab".
    """
    assert resolve_facet_option("AI", ["AI Lab", "AI Labs Brasil"]) is None
    assert resolve_facet_option("AI", ["AI"]) == (0, "AI")


def test_prefix_rule_takes_the_first_qualifying_prefix_in_order():
    resolved = resolve_facet_option(
        "Nubank Pagamentos SA", ["Nubank Pagamentos", "Nubank"]
    )
    assert resolved == (0, "Nubank Pagamentos")


def test_no_match_is_a_miss():
    assert resolve_facet_option("Stripe", ["Striped Fabrics", "Strip Mall"]) is None
    assert resolve_facet_option("Stripe", []) is None


def test_exact_match_is_diacritic_insensitive():
    resolved = resolve_facet_option("Itau Unibanco", ["Itaú Unibanco"])
    assert resolved == (0, "Itaú Unibanco")


def test_a_stripped_suffix_row_is_not_this_requests_exact_row():
    """The exact scan is folded equality, and the fold erased the suffix.

    "C++ Engineer" folded to "c engineer", so a request for "C Engineer" took
    the C++ row as an EXACT match — which also bypasses the repaint wait,
    because an exact row is taken from any list. The real row sat underneath
    and was never considered.
    """
    assert resolve_facet_option("C Engineer", ["C++ Engineer", "C Engineer"]) == (
        1,
        "C Engineer",
    )
    assert resolve_facet_option("NET Developer", [".NET Developer"]) is None
    assert resolve_facet_option("Staff Engineer", ["Staff+ Engineer"]) is None


def test_blank_request_never_takes_an_option():
    assert resolve_facet_option("   ", ["Anything"]) is None


def test_multiline_option_is_matched_on_its_first_line():
    """``all_inner_texts`` can carry a secondary line; the entity is line one."""
    resolved = resolve_facet_option(
        "Mercado Livre", ["Mercado Livre Brasil\n12,000 employees"]
    )
    assert resolved == (0, "Mercado Livre Brasil")


# ---------------------------------------------------------------------------
# The substitution carried through the receipt
# ---------------------------------------------------------------------------


def test_facet_target_landed_returns_the_landed_spelling():
    receipt = {
        "unapplied": {"companies": ["Tess AI"], "titles": []},
        "substituted": {
            "companies": {"Mercado Livre": "Mercado Livre Brasil"},
            "titles": {},
        },
    }
    assert facet_target_landed(
        {"Mercado Livre", "Tess AI", "Stripe"}, receipt, "companies"
    ) == {"Mercado Livre Brasil", "Stripe"}


def test_facet_target_landed_is_unchanged_without_substitutions():
    receipt = {"unapplied": {"companies": ["Tess AI"]}}
    assert facet_target_landed({"Stripe", "Tess AI"}, receipt, "companies") == {
        "Stripe"
    }


def test_substituted_console_line_is_one_prebuilt_line():
    receipt = {
        "substituted": {
            "companies": {"Mercado Livre": "Mercado Livre Brasil"},
            "titles": {},
        }
    }
    line = format_facet_substituted_console_line(19, receipt)
    assert line == (
        '  [facet] string #19 near-miss: companies "Mercado Livre"'
        '→"Mercado Livre Brasil"'
    )
    assert "\n" not in line


def test_substituted_console_line_is_silent_when_every_name_matched():
    assert format_facet_substituted_console_line(19, {}) is None
    assert (
        format_facet_substituted_console_line(
            19, {"substituted": {"companies": {}, "titles": {}}}
        )
        is None
    )


# ---------------------------------------------------------------------------
# The persisted resolution cache (a projection, never canonical state)
# ---------------------------------------------------------------------------


def test_cache_round_trips_by_normalized_request(tmp_path):
    write_facet_name_resolution(
        tmp_path, "companies", "Mercado  LIVRE", "Mercado Livre Brasil"
    )
    cache = read_facet_name_resolutions(tmp_path)
    assert cache["companies"]["mercado livre"] == "Mercado Livre Brasil"
    assert cache["titles"] == {}


def test_cache_write_preserves_other_dimensions_and_values(tmp_path):
    write_facet_name_resolution(tmp_path, "companies", "Tess AI", "Tess")
    write_facet_name_resolution(tmp_path, "titles", "MLE", "Machine Learning Engineer")
    write_facet_name_resolution(tmp_path, "companies", "Itaú", "Itau Unibanco")

    cache = read_facet_name_resolutions(tmp_path)
    assert cache["companies"] == {
        "tess ai": "Tess",
        "itau": "Itau Unibanco",
    }
    assert cache["titles"] == {"mle": "Machine Learning Engineer"}


def test_a_corrupt_cache_file_reads_as_empty_and_is_rewritten(tmp_path):
    (tmp_path / "facet_name_resolutions.json").write_text("{not json", encoding="utf-8")
    assert read_facet_name_resolutions(tmp_path) == {"companies": {}, "titles": {}}

    write_facet_name_resolution(tmp_path, "companies", "Tess AI", "Tess")
    assert read_facet_name_resolutions(tmp_path)["companies"] == {"tess ai": "Tess"}


def test_a_missing_state_dir_reads_empty_and_never_raises(tmp_path):
    assert read_facet_name_resolutions(None) == {"companies": {}, "titles": {}}
    assert read_facet_name_resolutions(tmp_path / "nope") == {
        "companies": {},
        "titles": {},
    }
    write_facet_name_resolution(None, "companies", "Tess AI", "Tess")


def test_an_unwritable_state_dir_is_logged_never_raised(tmp_path, caplog):
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory", encoding="utf-8")
    with caplog.at_level("WARNING"):
        write_facet_name_resolution(blocked, "companies", "Tess AI", "Tess")
    assert "facet_name_resolutions" in caplog.text
    assert read_facet_name_resolutions(blocked) == {"companies": {}, "titles": {}}


def test_the_sidecar_is_json_shaped_for_an_operator(tmp_path):
    write_facet_name_resolution(
        tmp_path, "companies", "Mercado Livre", "Mercado Livre Brasil"
    )
    payload = json.loads(
        (tmp_path / "facet_name_resolutions.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "companies": {"mercado livre": "Mercado Livre Brasil"},
        "titles": {},
    }
