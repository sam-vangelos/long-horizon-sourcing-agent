"""Tests for the vertical-agnostic calibration vocabulary (Slice 1).

Covers the new ``shared.brief_schema.Brief`` fields, the compat-Brief mirror
on ``shared.brief_loader.Brief``, the inert rendering helpers, and the Stage 0
warning-only validator wired in ``shared.brief_loader._load_v2_brief``.

These tests are deliberately schema-only. They do not touch
``linkedin/strategy.py``, ``linkedin/judgment_templates.py``,
``shared/judger.py``, or ``shared/search_memory.py`` — those consumers are
Slice 2's responsibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from shared.brief_loader import (
    Brief as CompatBrief,
    _load_v2_brief,
    _validate_v2_calibration,
    load_brief,
)
from shared.brief_schema import (
    AbbreviationCollision,
    BlacklistCategory,
    Brief as V2Brief,
    DomainLaneHint,
    ExampleCompound,
    TransferabilityExample,
)


ROOT = Path(__file__).parent.parent
LEGACY_BRIEF_PATH = ROOT / "config" / "FDL-Brazil" / "brief-brazil-real.json"


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _minimal_v2_raw() -> dict:
    """Minimum viable V2 brief raw dict (no calibration fields populated)."""
    return {
        "role_title": "Senior Marketing Lead",
        "role_level": "L5",
        "role_summary": "Generic vertical-agnostic test role.",
        "geography": "New York",
        "linkedin_project": "Test Project",
        "capability_areas": [
            {
                "name": "Capability A",
                "description": "Owns campaign launches.",
                "builder_signals": ["launched campaign"],
                "user_signals": ["read campaign reports"],
                "key_terms": ["launch"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns the launch.",
            "user_definition": "Watches the launch.",
            "edge_case_guidance": "Defer to evidence.",
        },
        "non_fit_patterns": [],
        "employer_signal_rules": [],
        "minimum_years_experience": 5,
        "minimum_bar_description": "Five years of launch ownership.",
        "facial_calibration": {
            "expected_yes_rate_low": 0.25,
            "expected_yes_rate_high": 0.55,
            "fast_exit_patterns": [],
            "trajectory_yes_patterns": [],
            "trajectory_ambiguous_patterns": [],
            "trajectory_no_patterns": [],
        },
    }


def _partially_populated_v2_raw() -> dict:
    """V2 brief raw dict claiming calibration provenance but incomplete.

    Exactly one calibration field is populated (``domain_verbs``) — this
    is the "hand-authored, left gaps" shape P9.5 must still WARN on,
    distinct from the "no intake producer emits these" all-absent shape.
    """
    raw = _minimal_v2_raw()
    raw["domain_verbs"] = ["owned", "launched"]
    return raw


def _populated_v2_raw() -> dict:
    """V2 brief raw dict with every Slice 1 calibration field populated."""
    raw = _minimal_v2_raw()
    raw.update({
        "role_level": "L7",
        "domain_verbs": ["owned", "launched", "scaled"],
        "domain_depth_objects": [
            "campaigns with measurable lift",
            "budget ownership over $10M",
        ],
        "transferability_examples": [
            {
                "result": "transfers",
                "source_context": "Indie distribution marketing",
                "target_context": "Studio marketing leadership",
                "rationale": "Channel strategy and campaign measurement transfer cleanly.",
            },
            {
                "result": "does_not_transfer",
                "source_context": "Consumer influencer marketing",
                "target_context": "Film slate marketing",
                "rationale": "Audience growth alone does not demonstrate launch orchestration.",
            },
        ],
        "canonical_framework_patterns": ["release planning", "P&A allocation"],
        "canonical_company_patterns": ["Northwind", "Neon", "Searchlight"],
        "canonical_title_patterns": ["Head of Marketing", "VP Marketing"],
        "canonical_broad_patterns": ["film marketing", "release campaign"],
        "edge_case_patterns": ["festival programming", "indie distribution"],
        "edge_case_company_patterns": ["Festival X", "Festival Y"],
        "sequencing_heuristics": "Lead with senior title; backload festival language.",
        "term_blacklist_categories": [
            {
                "label": "viewer-side language",
                "rationale": "Describes fandom, not operator-side work.",
                "terms": ["fan community", "movie buff"],
            }
        ],
        "abbreviation_collisions": [
            {
                "abbreviation": "P&A",
                "expansion": "prints and advertising",
                "standalone_allowed": False,
                "note": "Pair with expansion in mixed geographies.",
            }
        ],
        "example_compounds": [
            {
                "boolean": '("head of marketing") AND (film OR studio)',
                "purpose": "broad recall",
                "novelty_bucket": "canonical",
            }
        ],
        "domain_lane_hints": [
            {
                "lane": "distribution",
                "patterns": ["distribution", "release", "theatrical"],
            }
        ],
    })
    return raw


# ---------------------------------------------------------------------------
# Schema dataclass shape
# ---------------------------------------------------------------------------


def test_v2_brief_has_calibration_fields_with_safe_defaults():
    """All new calibration fields default to empty / "" without forcing args."""
    raw = _minimal_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief
    assert isinstance(nb, V2Brief)

    list_defaults = [
        "domain_verbs",
        "domain_depth_objects",
        "transferability_examples",
        "canonical_framework_patterns",
        "canonical_company_patterns",
        "canonical_title_patterns",
        "canonical_broad_patterns",
        "edge_case_patterns",
        "edge_case_company_patterns",
        "term_blacklist_categories",
        "abbreviation_collisions",
        "example_compounds",
        "domain_lane_hints",
    ]
    for name in list_defaults:
        assert getattr(nb, name) == [], f"{name} did not default to []"
    assert nb.sequencing_heuristics == ""


def test_v2_brief_helper_dataclasses_are_constructible():
    """The five new helper dataclasses build with their declared signatures."""
    te = TransferabilityExample(
        result="transfers",
        source_context="A",
        target_context="B",
        rationale="why",
    )
    assert te.result == "transfers"

    bc = BlacklistCategory(label="x", rationale="y", terms=["a"])
    assert bc.terms == ["a"]

    ac = AbbreviationCollision(abbreviation="P&A", expansion="prints and advertising")
    assert ac.standalone_allowed is False
    assert ac.note == ""

    ec = ExampleCompound(boolean='"x"', purpose="broad")
    assert ec.novelty_bucket == ""

    dl = DomainLaneHint(lane="distribution", patterns=["release"])
    assert dl.patterns == ["release"]


# ---------------------------------------------------------------------------
# Loader hydration
# ---------------------------------------------------------------------------


def test_v2_loader_hydrates_calibration_onto_both_briefs():
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief

    # _new_brief gets the full set.
    assert nb.domain_verbs == ["owned", "launched", "scaled"]
    assert nb.domain_depth_objects[0] == "campaigns with measurable lift"
    assert isinstance(nb.transferability_examples[0], TransferabilityExample)
    assert nb.transferability_examples[0].result == "transfers"
    assert nb.canonical_framework_patterns == ["release planning", "P&A allocation"]
    assert nb.canonical_company_patterns == ["Northwind", "Neon", "Searchlight"]
    assert nb.canonical_title_patterns == ["Head of Marketing", "VP Marketing"]
    assert nb.canonical_broad_patterns == ["film marketing", "release campaign"]
    assert nb.edge_case_patterns == ["festival programming", "indie distribution"]
    assert nb.edge_case_company_patterns == ["Festival X", "Festival Y"]
    assert nb.sequencing_heuristics.startswith("Lead with senior title")
    assert isinstance(nb.term_blacklist_categories[0], BlacklistCategory)
    assert nb.term_blacklist_categories[0].terms == ["fan community", "movie buff"]
    assert isinstance(nb.abbreviation_collisions[0], AbbreviationCollision)
    assert nb.abbreviation_collisions[0].abbreviation == "P&A"
    assert nb.abbreviation_collisions[0].standalone_allowed is False
    assert isinstance(nb.example_compounds[0], ExampleCompound)
    assert nb.example_compounds[0].novelty_bucket == "canonical"
    assert isinstance(nb.domain_lane_hints[0], DomainLaneHint)
    assert nb.domain_lane_hints[0].patterns == ["distribution", "release", "theatrical"]

    # Compat brief mirrors the strategy-relevant set.
    assert compat.domain_verbs == nb.domain_verbs
    assert compat.domain_depth_objects == nb.domain_depth_objects
    assert compat.transferability_examples == nb.transferability_examples
    assert compat.canonical_framework_patterns == nb.canonical_framework_patterns
    assert compat.canonical_company_patterns == nb.canonical_company_patterns
    assert compat.canonical_title_patterns == nb.canonical_title_patterns
    assert compat.canonical_broad_patterns == nb.canonical_broad_patterns
    assert compat.edge_case_patterns == nb.edge_case_patterns
    assert compat.edge_case_company_patterns == nb.edge_case_company_patterns
    assert compat.sequencing_heuristics == nb.sequencing_heuristics
    assert compat.term_blacklist_categories == nb.term_blacklist_categories
    assert compat.abbreviation_collisions == nb.abbreviation_collisions
    assert compat.example_compounds == nb.example_compounds
    assert compat.domain_lane_hints == nb.domain_lane_hints


def test_compat_brief_calibration_does_not_collide_with_existing_attrs():
    """Mirror set must not shadow pre-existing compat-Brief attributes."""
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    # The mirror added new names — pre-existing attrs must still hold their old values.
    assert compat.additional_search_terms == []
    assert compat.search_priorities == []
    assert isinstance(compat.key_terms_by_area, dict)
    # The compat brief still has employer_signal_rules-equivalent surface via raw,
    # but it does NOT have an attribute literally called employer_signal_rules.
    assert not hasattr(compat, "employer_signal_rules"), (
        "compat Brief should not gain an employer_signal_rules attribute via this slice"
    )


def test_v2_loader_defaults_calibration_fields_when_absent():
    """When the V2 brief omits the new fields, defaults are empty (no AI defaults)."""
    raw = _minimal_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief

    assert nb.domain_verbs == []
    assert nb.transferability_examples == []
    assert nb.canonical_broad_patterns == []
    assert nb.term_blacklist_categories == []
    assert nb.abbreviation_collisions == []
    assert nb.example_compounds == []
    assert nb.domain_lane_hints == []
    assert nb.sequencing_heuristics == ""

    # And the compat brief mirrors those empties.
    assert compat.domain_verbs == []
    assert compat.canonical_broad_patterns == []
    assert compat.term_blacklist_categories == []
    assert compat.sequencing_heuristics == ""


# ---------------------------------------------------------------------------
# Stage-0 validator behavior
# ---------------------------------------------------------------------------


def test_v2_loader_info_logs_when_no_calibration_fields_present(caplog):
    """P9.5: an intake-born brief (NO calibration fields at all — the
    ordinary shape, since no intake producer emits them) gets a single
    INFO line, never a WARNING and never a hard-fail threat.
    """
    raw = _minimal_v2_raw()
    with caplog.at_level(logging.INFO, logger="shared.brief_loader"):
        compat = _load_v2_brief(raw)

    warnings = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING and "calibration fields" in rec.getMessage()
    ]
    assert warnings == [], "all-fields-absent must not warn — that's the intake-born shape"

    infos = [
        rec for rec in caplog.records
        if rec.levelno == logging.INFO and "calibration fields" in rec.getMessage()
    ]
    assert len(infos) == 1, "expected exactly one summary calibration info line"
    msg = infos[0].getMessage()
    assert "hard-fail" not in msg.lower()
    # Each missing required calibration field should appear in the summary.
    for required in (
        "domain_verbs",
        "domain_depth_objects",
        "transferability_examples",
        "canonical_framework_patterns",
        "canonical_company_patterns",
        "canonical_title_patterns",
        "canonical_broad_patterns",
        "edge_case_patterns",
        "edge_case_company_patterns",
        "sequencing_heuristics",
        "term_blacklist_categories",
        "abbreviation_collisions",
        "example_compounds",
    ):
        assert required in msg, f"info summary missing field {required!r}"
    # Compat brief still loaded successfully.
    assert isinstance(compat, CompatBrief)


def test_v2_loader_warns_when_calibration_partially_populated(caplog):
    """P9.5: a brief that HAS at least one calibration field (claiming
    calibration provenance) still WARNS about the rest that are missing.
    This is the real omission case the Stage-0 validator exists for.
    """
    raw = _partially_populated_v2_raw()
    with caplog.at_level(logging.WARNING, logger="shared.brief_loader"):
        _load_v2_brief(raw)

    warnings = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING and "calibration fields" in rec.getMessage()
    ]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "hard-fail" not in msg.lower()
    # The one populated field must not appear as missing.
    assert "domain_verbs" not in msg
    # An unpopulated field from the same claimed-provenance brief must warn.
    assert "domain_depth_objects" in msg


def test_v2_loader_does_not_warn_when_calibration_fully_populated(caplog):
    """A fully-populated V2 brief produces no calibration warning or info."""
    raw = _populated_v2_raw()
    with caplog.at_level(logging.INFO, logger="shared.brief_loader"):
        _load_v2_brief(raw)
    cal_records = [
        rec for rec in caplog.records
        if rec.levelno in (logging.WARNING, logging.INFO)
        and "calibration fields" in rec.getMessage()
    ]
    assert cal_records == []


def test_validator_does_not_raise_on_missing_fields():
    """Stage-0 is warning/info-only — must NEVER raise."""
    raw = _minimal_v2_raw()
    compat = _load_v2_brief(raw)
    # Calling the validator directly on the missing-field brief still must not raise.
    _validate_v2_calibration(compat._new_brief, compat.id)


@pytest.mark.skipif(
    not LEGACY_BRIEF_PATH.is_file(),
    reason="Optional legacy Brazil brief JSON not under config/",
)
def test_legacy_brief_load_does_not_emit_calibration_warning(caplog):
    """Old-format briefs must NOT route through the V2 calibration validator."""
    with caplog.at_level(logging.WARNING, logger="shared.brief_loader"):
        legacy = load_brief(str(LEGACY_BRIEF_PATH))
    cal_warnings = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING and "calibration fields" in rec.getMessage()
    ]
    assert cal_warnings == [], (
        "legacy normalize_brief() path must stay silent on the calibration validator"
    )
    assert legacy.has_v2_schema is False
    # Legacy briefs do gain the new mirror attributes (defaults), but they are unused
    # and unset — they must not be hydrated by normalize_brief().
    assert legacy.domain_verbs == []
    assert legacy.canonical_broad_patterns == []
    assert legacy.sequencing_heuristics == ""


# ---------------------------------------------------------------------------
# Rendering helpers — populated and empty inputs
# ---------------------------------------------------------------------------


def _v2_with_calibration() -> V2Brief:
    return _load_v2_brief(_populated_v2_raw())._new_brief


def _v2_empty_calibration() -> V2Brief:
    return _load_v2_brief(_minimal_v2_raw())._new_brief


def test_domain_verbs_block_renders_when_populated():
    nb = _v2_with_calibration()
    out = nb.domain_verbs_block()
    assert "owned" in out and "launched" in out and "scaled" in out


def test_domain_verbs_block_empty_for_empty_input():
    assert _v2_empty_calibration().domain_verbs_block() == ""


def test_domain_depth_objects_block_renders_bullets():
    nb = _v2_with_calibration()
    out = nb.domain_depth_objects_block()
    assert out.startswith("- campaigns with measurable lift")
    assert "budget ownership over $10M" in out


def test_domain_depth_objects_block_empty_for_empty_input():
    assert _v2_empty_calibration().domain_depth_objects_block() == ""


def test_transferability_examples_block_unfiltered_renders_both_results():
    nb = _v2_with_calibration()
    out = nb.transferability_examples_block()
    assert "[transfers]" in out
    assert "[does_not_transfer]" in out
    assert "Indie distribution marketing" in out
    assert "Rationale:" in out


def test_transferability_examples_block_filtered_by_result():
    nb = _v2_with_calibration()
    transfers_only = nb.transferability_examples_block(result="transfers")
    nontransfers_only = nb.transferability_examples_block(result="does_not_transfer")
    assert "Indie distribution marketing" in transfers_only
    assert "Consumer influencer marketing" not in transfers_only
    assert "Consumer influencer marketing" in nontransfers_only
    assert "Indie distribution marketing" not in nontransfers_only


def test_transferability_examples_block_empty_for_empty_input():
    nb = _v2_empty_calibration()
    assert nb.transferability_examples_block() == ""
    assert nb.transferability_examples_block(result="transfers") == ""


def test_term_blacklist_block_renders_with_terms():
    nb = _v2_with_calibration()
    out = nb.term_blacklist_block()
    assert "viewer-side language" in out
    assert "fan community" in out
    assert "Describes fandom" in out


def test_term_blacklist_block_empty_for_empty_input():
    assert _v2_empty_calibration().term_blacklist_block() == ""


def test_abbreviation_collisions_block_renders():
    nb = _v2_with_calibration()
    out = nb.abbreviation_collisions_block()
    assert "P&A" in out
    assert "prints and advertising" in out
    assert "pair with expansion" in out
    assert "Note:" in out


def test_abbreviation_collisions_block_empty_for_empty_input():
    assert _v2_empty_calibration().abbreviation_collisions_block() == ""


def test_example_compounds_block_renders_purpose_and_bucket():
    nb = _v2_with_calibration()
    out = nb.example_compounds_block()
    assert "broad recall" in out
    assert "[canonical]" in out
    assert "head of marketing" in out


def test_example_compounds_block_empty_for_empty_input():
    assert _v2_empty_calibration().example_compounds_block() == ""


def test_domain_lane_hints_map_returns_dict():
    nb = _v2_with_calibration()
    out = nb.domain_lane_hints_map()
    assert out == {"distribution": ["distribution", "release", "theatrical"]}


def test_domain_lane_hints_map_empty_for_empty_input():
    assert _v2_empty_calibration().domain_lane_hints_map() == {}


def test_strategy_pattern_sets_returns_full_pattern_dict():
    nb = _v2_with_calibration()
    sets = nb.strategy_pattern_sets()
    assert set(sets.keys()) == {
        "canonical_framework",
        "canonical_company",
        "canonical_title",
        "canonical_broad",
        "edge_case",
        "edge_case_company",
    }
    assert sets["canonical_broad"] == ["film marketing", "release campaign"]
    assert sets["edge_case"] == ["festival programming", "indie distribution"]


def test_strategy_pattern_sets_returns_empty_lists_when_unset():
    nb = _v2_empty_calibration()
    sets = nb.strategy_pattern_sets()
    assert sets == {
        "canonical_framework": [],
        "canonical_company": [],
        "canonical_title": [],
        "canonical_broad": [],
        "edge_case": [],
        "edge_case_company": [],
    }


# ---------------------------------------------------------------------------
# Compat-bridge aliasing — mirrored calibration fields must be detached copies
# ---------------------------------------------------------------------------
#
# Every mirrored calibration list on the compat ``Brief`` must be value-equal
# to the corresponding field on ``_new_brief`` but must not be the SAME object.
# Mutating either side must not leak into the other. ``sequencing_heuristics``
# is a plain str and intentionally immutable, so no identity check applies.

_MIRRORED_LIST_FIELDS = (
    # Plain string lists.
    "domain_verbs",
    "domain_depth_objects",
    "canonical_framework_patterns",
    "canonical_company_patterns",
    "canonical_title_patterns",
    "canonical_broad_patterns",
    "edge_case_patterns",
    "edge_case_company_patterns",
    # Dataclass lists.
    "transferability_examples",
    "term_blacklist_categories",
    "abbreviation_collisions",
    "example_compounds",
    "domain_lane_hints",
)


def test_compat_mirror_fields_are_value_equal_to_new_brief():
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief
    for name in _MIRRORED_LIST_FIELDS:
        assert getattr(compat, name) == getattr(nb, name), (
            f"compat.{name} drifted from _new_brief.{name} after detachment"
        )


def test_compat_mirror_lists_are_distinct_objects_from_new_brief():
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief
    for name in _MIRRORED_LIST_FIELDS:
        compat_val = getattr(compat, name)
        new_val = getattr(nb, name)
        assert compat_val is not new_val, (
            f"compat.{name} is the same object as _new_brief.{name}; "
            "compat mirror must hold a detached copy"
        )


def test_mutating_compat_string_list_does_not_leak_into_new_brief():
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief
    original = list(nb.canonical_framework_patterns)

    compat.canonical_framework_patterns.append("INJECTED")
    assert nb.canonical_framework_patterns == original
    assert "INJECTED" not in nb.canonical_framework_patterns

    compat.canonical_framework_patterns.clear()
    assert nb.canonical_framework_patterns == original


def test_mutating_compat_dataclass_list_does_not_leak_into_new_brief():
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief
    original_count = len(nb.term_blacklist_categories)

    compat.term_blacklist_categories.append(
        BlacklistCategory(label="injected", rationale="r", terms=["x"])
    )
    assert len(nb.term_blacklist_categories) == original_count


def test_mutating_dataclass_instance_inside_compat_list_does_not_leak():
    """Dataclass elements must be deep-copied, not shallow-copied."""
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief

    # Pin original state on _new_brief so any leak is detectable.
    original_label = nb.term_blacklist_categories[0].label
    original_terms = list(nb.term_blacklist_categories[0].terms)

    # Mutate the FIRST element on the compat side: scalar field + nested list.
    compat.term_blacklist_categories[0].label = "MUTATED_LABEL"
    compat.term_blacklist_categories[0].terms.append("MUTATED_TERM")

    # _new_brief's element must remain untouched.
    assert nb.term_blacklist_categories[0].label == original_label
    assert nb.term_blacklist_categories[0].terms == original_terms
    assert "MUTATED_TERM" not in nb.term_blacklist_categories[0].terms


def test_replacing_compat_list_entirely_does_not_affect_new_brief():
    raw = _populated_v2_raw()
    compat = _load_v2_brief(raw)
    nb = compat._new_brief
    original = list(nb.domain_lane_hints)

    compat.domain_lane_hints = []
    assert nb.domain_lane_hints == original
    assert nb.domain_lane_hints is not compat.domain_lane_hints
