"""The enricher behind its flag, in the two places plans are finalized.

The flag defaults off and the only thing that matters about that default is
that it is inert: a plan finalized with the flag off must serialize to
exactly what it serialized to before this existed. Everything else here is
about what the flag turns on.
"""

from __future__ import annotations

import json

import pytest
from pathlib import Path

import shared.config as config
from linkedin import boolean_enrichment
from linkedin import strategy
from linkedin.boolean_score.parser import parse
from linkedin.boolean_score.score import score
from shared.schemas import AdaptationResponse, ExecutionPlan


@pytest.fixture
def brief():
    from tests.test_lane_key_integrity import _make_brief

    return _make_brief()


def _plan() -> ExecutionPlan:
    return ExecutionPlan.from_dict(
        {
            "strategy_rationale": "t",
            "generated_strings": [
                {
                    "boolean": '("post-training" OR "RLHF")',
                    "rationale": "r",
                    "family_key": "f",
                    "domain_lane": "general",
                }
            ],
            "coverage_gaps": [
                {
                    "gap": "g",
                    "suggested_boolean": '("reward model" OR "DPO")',
                    "rationale": "r",
                }
            ],
        }
    )


@pytest.fixture(autouse=True)
def _forget_the_cached_lexicon():
    boolean_enrichment.load_role_lexicon.cache_clear()
    yield
    boolean_enrichment.load_role_lexicon.cache_clear()


# --- the flag is inert by default ------------------------------------------


@pytest.mark.structural
def test_the_flag_defaults_off():
    """The CODE default is off. ``shared.config`` auto-loads the operator's
    ``.env``, where the flag may legitimately be on, so this asserts the
    declared default rather than the loaded value."""
    source = Path(config.__file__).read_text()
    assert '_env_flag(\n    "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", "false"' in source or (
        '_env_flag("LINKEDIN_LEXICON_ENRICHMENT_ENABLED", "false"' in source
    )


def test_a_finalized_plan_is_byte_identical_with_the_flag_off(monkeypatch, brief):
    """The gate is proved by breaking what it gates.

    A pass-through enricher would make this test pass whether or not the flag
    is read, so the enricher is replaced with one that mangles every string.
    Flag off, the plan must come back untouched anyway.
    """
    monkeypatch.setattr(config, "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", False)
    monkeypatch.setattr(strategy._config, "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", False)

    def _mangle(boolean, lexicon, **kwargs):
        raise AssertionError("the enricher ran with the flag off")

    monkeypatch.setattr(boolean_enrichment, "enrich", _mangle)

    finalized = strategy._finalize_execution_plan(brief, _plan(), None)
    payload = finalized.to_dict()
    assert payload["generated_strings"][0]["boolean"] == '("post-training" OR "RLHF")'
    assert payload["coverage_gaps"][0]["suggested_boolean"] == (
        '("reward model" OR "DPO")'
    )
    blob = json.dumps(payload)
    for key in ("boolean_original", "boolean_enrichment", "boolean_score"):
        assert key not in blob


def test_the_flag_off_leaves_adaptation_strings_untouched(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", False)
    items = [{"boolean": '("post-training" OR "RLHF")', "rationale": "r"}]
    boolean_enrichment.enrich_items(items, "boolean")
    assert items == [{"boolean": '("post-training" OR "RLHF")', "rationale": "r"}]


# --- what the flag turns on -------------------------------------------------


def _enabled(monkeypatch) -> None:
    monkeypatch.setattr(config, "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(strategy._config, "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", True)


def test_the_flag_on_enriches_and_keeps_the_receipt(monkeypatch, brief):
    _enabled(monkeypatch)
    finalized = strategy._finalize_execution_plan(brief, _plan(), None)
    item = finalized.to_dict()["generated_strings"][0]

    assert item["boolean_original"] == '("post-training" OR "RLHF")'
    assert item["boolean"] != item["boolean_original"]
    assert parse(item["boolean"]).legal
    assert item["boolean_enrichment"]["unchanged"] is False
    assert item["boolean_enrichment"]["changes"]
    assert item["boolean_score"]["composite"] > 0

    before = score(item["boolean_original"], boolean_enrichment.load_role_lexicon())
    after = score(item["boolean"], boolean_enrichment.load_role_lexicon())
    assert after.log_composite >= before.log_composite


def test_the_flag_on_enriches_coverage_gaps(monkeypatch, brief):
    _enabled(monkeypatch)
    finalized = strategy._finalize_execution_plan(brief, _plan(), None)
    gap = finalized.to_dict()["coverage_gaps"][0]
    assert gap["suggested_boolean_original"] == '("reward model" OR "DPO")'
    assert parse(gap["suggested_boolean"]).legal
    assert "suggested_boolean_enrichment" in gap


def test_the_flag_on_enriches_adaptation_new_strings(monkeypatch, brief):
    _enabled(monkeypatch)
    adaptation = AdaptationResponse.from_dict(
        {
            "new_strings": [
                {
                    "boolean": '("post-training" OR "RLHF")',
                    "rationale": "r",
                    "family_key": "f",
                    "domain_lane": "general",
                }
            ]
        }
    )
    strategy._annotate_adaptation_metadata(brief, adaptation, None)
    item = adaptation.new_strings[0]
    assert item["boolean_original"] == '("post-training" OR "RLHF")'
    assert item["boolean"] != item["boolean_original"]
    assert "boolean_enrichment" in item


def test_enrichment_never_prints(monkeypatch, brief, capsys):
    """The run console is a fixed surface; enrichment is not on it."""
    _enabled(monkeypatch)
    strategy._finalize_execution_plan(brief, _plan(), None)
    noisy = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "enrich" in line.lower()
    ]
    assert noisy == []


def test_a_broken_lexicon_path_leaves_the_plan_alone(monkeypatch, brief):
    _enabled(monkeypatch)
    monkeypatch.setattr(config, "LINKEDIN_BOOLEAN_LEXICON_PATH", "/nope/missing.json")
    boolean_enrichment.load_role_lexicon.cache_clear()
    finalized = strategy._finalize_execution_plan(brief, _plan(), None)
    item = finalized.to_dict()["generated_strings"][0]
    assert item["boolean"] == '("post-training" OR "RLHF")'
    assert "boolean_original" not in item


def test_an_illegal_string_is_left_as_authored(monkeypatch):
    _enabled(monkeypatch)
    items = [{"boolean": '("unclosed OR "x")'}]
    boolean_enrichment.enrich_items(items, "boolean")
    assert items[0]["boolean"] == '("unclosed OR "x")'
    assert items[0]["boolean_enrichment"]["legal"] is False
