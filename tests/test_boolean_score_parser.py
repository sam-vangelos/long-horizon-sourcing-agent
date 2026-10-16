"""Parser legality and warning findings (plans/boolean-scorer.md §4).

Red-first: one case per legality error code, the two warning findings the
parser itself emits, the precedence contract, and a whole-corpus parse of the
76 real strings. Composite-0-on-illegal is asserted in test_boolean_score.py,
where the scorer lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linkedin.boolean_score.parser import And, Not, Or, Term, parse

FIXTURE = Path(__file__).parent / "fixtures" / "boolean_corpus" / "ml_runs_76.json"


def _error_codes(result) -> set[str]:
    return {error.code for error in result.errors}


def _finding_codes(result) -> list[str]:
    return [finding.code for finding in result.findings]


# --- legality errors: one test per code in §4 ------------------------------


@pytest.mark.parametrize("boolean", ["", "   ", "\t\n"])
def test_empty_input_is_illegal(boolean):
    result = parse(boolean)
    assert result.legal is False
    assert "empty_input" in _error_codes(result)


def test_unclosed_open_parenthesis():
    result = parse('("alpha" OR "beta"')
    assert result.legal is False
    assert "unbalanced_parenthesis" in _error_codes(result)


def test_unmatched_closing_parenthesis():
    result = parse('"alpha") OR "beta"')
    assert result.legal is False
    assert "unbalanced_parenthesis" in _error_codes(result)


def test_unclosed_quote():
    result = parse('("alpha" OR "beta)')
    assert result.legal is False
    assert "unclosed_quote" in _error_codes(result)


def test_double_ampersand_operator():
    result = parse('"alpha" && "beta"')
    assert result.legal is False
    assert "malformed_operator" in _error_codes(result)


def test_double_pipe_operator():
    result = parse('"alpha" || "beta"')
    assert result.legal is False
    assert "malformed_operator" in _error_codes(result)


def test_bare_ampersand_operator():
    result = parse('"alpha" & "beta"')
    assert result.legal is False
    assert "malformed_operator" in _error_codes(result)


def test_wildcard_outside_quotes():
    result = parse('("alpha" OR "beta") AND engineer*')
    assert result.legal is False
    assert "wildcard_operator" in _error_codes(result)


def test_wildcard_inside_quotes_is_not_an_error():
    result = parse('("engineer*" OR "beta")')
    assert result.legal is True


def test_empty_group():
    result = parse('("alpha") AND ()')
    assert result.legal is False
    assert "empty_group" in _error_codes(result)


def test_group_holding_only_an_operator():
    result = parse('("alpha") AND (OR)')
    assert result.legal is False
    assert "empty_group" in _error_codes(result)


@pytest.mark.parametrize(
    "boolean",
    [
        'AND "alpha"',
        '"alpha" OR',
        '("alpha" OR )',
        '( OR "alpha")',
        '"alpha" AND AND "beta"',
        'NOT',
        '"alpha" AND NOT',
    ],
)
def test_dangling_operator(boolean):
    result = parse(boolean)
    assert result.legal is False
    assert "dangling_operator" in _error_codes(result)


def test_not_over_and_bearing_group():
    result = parse('("alpha") NOT ("beta" AND "gamma")')
    assert result.legal is False
    assert "not_group_contains_and" in _error_codes(result)


def test_not_over_implicit_and_group():
    result = parse('("alpha") NOT ("beta" "gamma")')
    assert result.legal is False
    assert "not_group_contains_and" in _error_codes(result)


def test_not_over_pure_or_group_is_legal():
    result = parse('("alpha") NOT ("beta" OR "gamma")')
    assert result.legal is True


def test_error_carries_code_message_and_span():
    result = parse('"alpha" && "beta"')
    error = next(e for e in result.errors if e.code == "malformed_operator")
    assert error.message
    assert error.span == (8, 10)


# --- warnings, never errors -------------------------------------------------


@pytest.mark.parametrize("boolean", ['("alpha" or "beta")', '"alpha" and "beta"'])
def test_lowercase_operator_is_a_warning_not_an_error(boolean):
    result = parse(boolean)
    assert result.legal is True
    assert "lowercase_operator" in _finding_codes(result)
    assert all(finding.severity == "warning" for finding in result.findings)


def test_unquoted_multiword_run_is_one_finding_per_run():
    result = parse('("alpha" OR reward model shaping) AND ("beta" OR gamma)')
    assert result.legal is True
    assert _finding_codes(result).count("unquoted_multiword_run") == 1


def test_single_bare_token_is_not_a_multiword_run():
    result = parse('("alpha" OR beta)')
    assert "unquoted_multiword_run" not in _finding_codes(result)


def test_operator_words_inside_quotes_are_phrase_content():
    result = parse('("research and development" OR "beta")')
    assert result.legal is True
    assert _finding_codes(result) == []
    assert isinstance(result.tree, Or)
    assert result.tree.children[0] == Term("research and development", True)


# --- shape ------------------------------------------------------------------


def test_precedence_not_binds_tighter_than_and():
    tree = parse('NOT "alpha" AND "beta"').tree
    assert isinstance(tree, And)
    assert isinstance(tree.children[0], Not)
    assert tree.children[0].child == Term("alpha", True)


def test_precedence_and_binds_tighter_than_or():
    tree = parse('"alpha" OR "beta" AND "gamma"').tree
    assert isinstance(tree, Or)
    assert tree.children[0] == Term("alpha", True)
    assert isinstance(tree.children[1], And)


def test_adjacent_primaries_are_an_implicit_and():
    tree = parse('("alpha") ("beta")').tree
    assert isinstance(tree, And)
    assert len(tree.children) == 2


def test_nested_group_keeps_depth():
    result = parse('("alpha" OR ("beta" AND ("gamma" OR "delta")))')
    assert result.legal is True
    assert result.max_depth == 3


# --- the corpus -------------------------------------------------------------


def _corpus() -> list[dict]:
    return json.loads(FIXTURE.read_text())["strings"]


@pytest.mark.parametrize("entry", _corpus(), ids=lambda e: e["id"])
def test_every_corpus_string_parses(entry):
    result = parse(entry["boolean"])
    assert result.legal is True, [e.code for e in result.errors]
    assert result.tree is not None
