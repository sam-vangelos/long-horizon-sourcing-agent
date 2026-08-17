"""Tests for V2 brief hydration in `shared.brief_loader._load_v2_brief`.

Pins the contract Slice 1 of the executive-search module depends on:

- A V2 brief carrying `confidentiality_class` + `prior_search` +
  `board_signals` + `executive_movement_window_days` +
  `executive_calibration` round-trips through `load_brief()` into the
  structured `_new_brief` dataclass AND mirrors onto the compat
  `Brief` so non-V2 consumers can read the values without spelunking.
- A V2 brief WITHOUT those keys hydrates to the dataclass defaults
  (no behavior change; Slice 1's "no behavior changes" mandate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.brief_loader import load_brief
from shared.brief_schema import (
    BoardSignalRules,
    ExecutiveCalibration,
    MarketDensity,
    PriorSearchContext,
)


def _minimal_v2_brief() -> dict:
    return {
        "role_title": "VP Engineering",
        "role_summary": "Owns engineering org for a series-C company.",
        "geography": "United States",
        "linkedin_project": "exec-search-vp-eng",
        "minimum_years_experience": 12,
        "minimum_bar_description": "10+ years engineering leadership.",
        "engagement_context": {
            "hiring_company": "ExampleCo",
            "engagement_description": "A VP Engineering search.",
            "talent_bar_statement": "Organization-wide ownership clears the bar.",
            "selectivity_posture": "selective",
        },
        "capability_areas": [
            {
                "name": "Org leadership",
                "description": "Builds and runs 50+ person engineering orgs.",
                "builder_signals": ["VP-level scope", "headcount growth"],
                "user_signals": ["IC-level work primarily"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns engineering strategy + delivery.",
            "user_definition": "Manages individual teams without org-wide scope.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }


def _write_brief(tmp_path: Path, payload: dict) -> Path:
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(payload))
    return brief_path


def test_load_brief_round_trips_engagement_context_to_both_views(
    tmp_path: Path,
) -> None:
    payload = _minimal_v2_brief()

    brief = load_brief(_write_brief(tmp_path, payload))

    assert brief.engagement_context == payload["engagement_context"]
    assert brief._new_brief.engagement_context == payload["engagement_context"]
    assert brief.engagement_context is not brief._new_brief.engagement_context


@pytest.mark.parametrize(
    "legacy_context",
    [pytest.param(None, id="absent"), {}, {"hiring_company": "ExampleCo"}, "bad"],
)
def test_load_brief_missing_engagement_posture_warns_once_and_falls_back(
    tmp_path: Path,
    caplog,
    legacy_context,
) -> None:
    payload = _minimal_v2_brief()
    if legacy_context is None:
        payload.pop("engagement_context")
    else:
        payload["engagement_context"] = legacy_context

    with caplog.at_level("WARNING", logger="shared.brief_loader"):
        brief = load_brief(_write_brief(tmp_path, payload))

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "missing engagement_context selectivity posture" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "market-density compatibility posture" in warnings[0]
    expected = legacy_context if isinstance(legacy_context, dict) else {}
    assert brief.engagement_context == expected
    assert brief._new_brief.engagement_context == expected


@pytest.mark.parametrize(
    "raw_density,expected",
    [
        ("sparse", MarketDensity.SPARSE),
        ("moderate", MarketDensity.MODERATE),
        ("dense", MarketDensity.DENSE),
        ("unknown", MarketDensity.MODERATE),
        (None, MarketDensity.MODERATE),
        (42, MarketDensity.MODERATE),
    ],
)
def test_load_brief_coerces_unknown_legacy_density_to_selective_default(
    tmp_path: Path,
    raw_density,
    expected,
) -> None:
    payload = _minimal_v2_brief()
    payload["market_density"] = raw_density

    brief = load_brief(_write_brief(tmp_path, payload))

    assert brief._new_brief.market_density is expected


def test_load_brief_hydrates_default_exec_search_fields(tmp_path: Path) -> None:
    """A V2 brief without exec_search keys gets the dataclass defaults."""

    brief_path = _write_brief(tmp_path, _minimal_v2_brief())
    brief = load_brief(brief_path)

    assert brief.confidentiality_class == "open"
    assert isinstance(brief.prior_search, PriorSearchContext)
    assert brief.prior_search.ruled_out_urls == []
    assert brief.prior_search.ruled_out_notes == ""
    assert brief.prior_search.earlier_run_ids == []
    assert isinstance(brief.board_signals, BoardSignalRules)
    assert brief.board_signals.relevant_board_companies == []
    assert brief.board_signals.relevant_executive_alumni_companies == []
    assert brief.executive_movement_window_days == 180
    assert brief.executive_calibration is None


def test_load_brief_hydrates_full_exec_search_fields(tmp_path: Path) -> None:
    """A V2 brief carrying every exec_search key hydrates onto compat Brief AND _new_brief."""

    payload = _minimal_v2_brief()
    payload["confidentiality_class"] = "blind"
    payload["prior_search"] = {
        "ruled_out_urls": [
            "https://linkedin.com/in/cand-a",
            "https://linkedin.com/in/cand-b",
        ],
        "ruled_out_notes": "Both passed in 2024 search; client moved on.",
        "earlier_run_ids": ["run_2024_q3"],
    }
    payload["board_signals"] = {
        "relevant_board_companies": ["AcmeCorp", "BetaInc"],
        "relevant_executive_alumni_companies": ["AlphaCo"],
        "adjacency_rationale": "Client board has 2 AcmeCorp alums.",
    }
    payload["board_signals"]
    payload["executive_movement_window_days"] = 90
    payload["executive_calibration"] = {
        "sector": "Healthcare",
        "stage": "Series D",
        "pnl_scale_usd": "$200M ARR",
        "register_notes": "Operator-builder bias.",
    }

    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    # Compat Brief mirror.
    assert brief.confidentiality_class == "blind"
    assert brief.prior_search.ruled_out_urls == [
        "https://linkedin.com/in/cand-a",
        "https://linkedin.com/in/cand-b",
    ]
    assert brief.prior_search.ruled_out_notes == (
        "Both passed in 2024 search; client moved on."
    )
    assert brief.prior_search.earlier_run_ids == ["run_2024_q3"]
    assert brief.board_signals.relevant_board_companies == ["AcmeCorp", "BetaInc"]
    assert brief.board_signals.relevant_executive_alumni_companies == ["AlphaCo"]
    assert brief.board_signals.adjacency_rationale == (
        "Client board has 2 AcmeCorp alums."
    )
    assert brief.executive_movement_window_days == 90
    assert isinstance(brief.executive_calibration, ExecutiveCalibration)
    assert brief.executive_calibration.sector == "Healthcare"
    assert brief.executive_calibration.stage == "Series D"
    assert brief.executive_calibration.pnl_scale_usd == "$200M ARR"
    assert brief.executive_calibration.register_notes == "Operator-builder bias."

    # Structured _new_brief carries the same.
    assert brief.has_v2_schema
    new_brief = brief._new_brief
    assert new_brief.confidentiality_class == "blind"
    assert new_brief.prior_search.ruled_out_urls == [
        "https://linkedin.com/in/cand-a",
        "https://linkedin.com/in/cand-b",
    ]
    assert new_brief.board_signals.relevant_board_companies == ["AcmeCorp", "BetaInc"]
    assert new_brief.executive_movement_window_days == 90
    assert isinstance(new_brief.executive_calibration, ExecutiveCalibration)


def test_load_brief_tolerates_malformed_exec_search_blocks(tmp_path: Path) -> None:
    """Defensive coercion: bad shapes degrade to defaults without crashing."""

    payload = _minimal_v2_brief()
    payload["prior_search"] = "not a dict"  # garbage
    payload["board_signals"] = ["also wrong"]
    payload["executive_calibration"] = "should be a dict"
    payload["executive_movement_window_days"] = "not an int"

    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    assert brief.prior_search.ruled_out_urls == []
    assert brief.board_signals.relevant_board_companies == []
    assert brief.executive_calibration is None
    assert brief.executive_movement_window_days == 180


def test_load_brief_compat_mirror_isolated_from_new_brief(tmp_path: Path) -> None:
    """`prior_search` mirror on compat Brief must not share mutable state with `_new_brief`.

    Mirrors the `_detach` pattern used for vertical-agnostic calibration
    fields. Mutating the compat Brief's lists must not affect the
    structured `_new_brief`.
    """

    payload = _minimal_v2_brief()
    payload["prior_search"] = {"ruled_out_urls": ["a", "b"]}

    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    brief.prior_search.ruled_out_urls.append("c")

    assert brief._new_brief.prior_search.ruled_out_urls == ["a", "b"]


# ---------------------------------------------------------------------------
# OSS Maintainers module Slice 2 — V2 brief hydration
# ---------------------------------------------------------------------------


def test_load_brief_hydrates_default_oss_maintainer_fields(tmp_path: Path) -> None:
    """A V2 brief without OSS Maintainer keys gets the dataclass defaults.

    Behavior-preserving for classic github briefs per spec §11:
    `target_projects` empty ⇒ classifier and full-eval block are
    no-ops in Slice 6.
    """

    brief_path = _write_brief(tmp_path, _minimal_v2_brief())
    brief = load_brief(brief_path)

    assert brief.target_projects == []
    assert brief.target_stacks == []
    assert brief.maintainership_level == "contributor"
    assert brief._new_brief.target_projects == []
    assert brief._new_brief.target_stacks == []
    assert brief._new_brief.maintainership_level == "contributor"


def test_load_brief_hydrates_full_oss_maintainer_fields(tmp_path: Path) -> None:
    """A V2 brief carrying every OSS Maintainer key hydrates onto compat AND _new_brief."""

    payload = _minimal_v2_brief()
    payload["target_projects"] = ["kubernetes/kubernetes", "etcd-io/etcd"]
    payload["target_stacks"] = ["go", "container-orchestration"]
    payload["maintainership_level"] = "maintainer"

    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    assert brief.target_projects == ["kubernetes/kubernetes", "etcd-io/etcd"]
    assert brief.target_stacks == ["go", "container-orchestration"]
    assert brief.maintainership_level == "maintainer"

    assert brief._new_brief.target_projects == [
        "kubernetes/kubernetes",
        "etcd-io/etcd",
    ]
    assert brief._new_brief.target_stacks == ["go", "container-orchestration"]
    assert brief._new_brief.maintainership_level == "maintainer"


def test_load_brief_oss_maintainer_compat_mirror_isolated_from_new_brief(
    tmp_path: Path,
) -> None:
    """`target_projects` mirror on compat Brief must not share mutable state.

    Mirrors the `_detach` pattern used for vertical-agnostic
    calibration fields and exec_search blocks. Mutating the compat
    Brief's lists must not affect the structured `_new_brief`.
    """

    payload = _minimal_v2_brief()
    payload["target_projects"] = ["kubernetes/kubernetes", "rust-lang/rust"]

    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    brief.target_projects.append("etcd-io/etcd")

    assert brief._new_brief.target_projects == [
        "kubernetes/kubernetes",
        "rust-lang/rust",
    ]


def test_load_brief_filters_malformed_oss_maintainer_entries(
    tmp_path: Path,
) -> None:
    """Defensive coercion: non-string list entries drop, garbage levels degrade."""

    payload = _minimal_v2_brief()
    # Intentionally seed garbage; validate_v2_brief WOULD reject this,
    # but the loader's defensive coercion should still produce a sane
    # Brief if it ever bypasses validation (e.g., legacy raw load).
    payload["target_projects"] = ["kubernetes/kubernetes", None, ""]
    payload["target_stacks"] = ["go"]
    payload["maintainership_level"] = ""

    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    assert brief.target_projects == ["kubernetes/kubernetes"]
    assert brief.target_stacks == ["go"]
    # Empty string degrades to default per loader coercion.
    assert brief.maintainership_level == "contributor"


# ---------------------------------------------------------------------------
# Intake shape tolerance — non_fit_patterns emitted as an array of strings
# ---------------------------------------------------------------------------


def test_load_v2_brief_tolerates_string_shaped_non_fit_patterns(
    tmp_path: Path,
) -> None:
    """The conversational extractor emits `non_fit_patterns` as short strings.

    `extractor.py` instructs the LLM to emit `non_fit_patterns` as an array
    of short strings; `merge_extracted` wholesale-replaces with no coercion;
    `validate_v2_brief` has no `non_fit_patterns` branch so the string shape
    passes validation; `intake.py` then writes that string-shaped draft to
    `config/<slug>/brief.json`. The canonical loader must tolerate BOTH wire
    shapes (composer emits dicts, extractor emits strings) rather than
    crashing with `TypeError: string indices must be integers` when it
    subscripts a `str` as if it were a dict.

    Pre-fix this raises `TypeError` at `brief_loader.py:256` (`nf["label"]`).
    Post-fix the string normalizes to `{"label": <str>, "why_not": <str>}`
    and feeds BOTH subscript sites (the `NonFitPattern` build and the
    `noise_archetypes` mapping) without a crash.
    """

    payload = _minimal_v2_brief()
    # Exactly the shape the extractor emits: an array of short strings.
    payload["non_fit_patterns"] = [
        "Title without ownership",
        "AI adjacency only",
    ]

    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    # Structured _new_brief: string normalized into a NonFitPattern where
    # label and why_not both fall back to the raw string.
    new_brief = brief._new_brief
    assert new_brief is not None
    assert len(new_brief.non_fit_patterns) == 2
    assert new_brief.non_fit_patterns[0].label == "Title without ownership"
    assert new_brief.non_fit_patterns[0].why_not == "Title without ownership"
    assert new_brief.non_fit_patterns[1].label == "AI adjacency only"

    # The SECOND subscript site (non_fit_patterns → noise_archetypes on the
    # compat Brief) must be exercised from the same normalized list and must
    # not crash on the string shape.
    assert brief.noise_archetypes[0]["name"] == "Title without ownership"
    assert brief.noise_archetypes[0]["description"] == "Title without ownership"
    assert brief.noise_archetypes[1]["name"] == "AI adjacency only"


# ---------------------------------------------------------------------------
# Preflight v2 structured geography -> permanent_filters (Codex review, Wave 1)
# ---------------------------------------------------------------------------


def _codex_geo_raw(geography):
    from tests.test_calibration_brief_fields import _minimal_v2_raw

    raw = _minimal_v2_raw()
    if geography is not None:
        raw["geography"] = geography
    else:
        raw.pop("geography", None)
    return raw


def test_v2_loader_joins_structured_geography_facet_candidates():
    from shared.brief_loader import _load_v2_brief

    brief = _load_v2_brief(
        _codex_geo_raw({"facet_candidates": ["Colombia", "Brazil"], "rationale": "JD"})
    )
    assert brief.permanent_filters["Location"] == "Colombia; Brazil"


def test_v2_loader_empty_facet_candidates_sets_no_location():
    """{'facet_candidates': [], 'rationale': ''} means the JD states no
    geography — it must NOT become a stringified-dict Location facet that
    trips the fail-closed geography gate."""
    from shared.brief_loader import _load_v2_brief

    brief = _load_v2_brief(_codex_geo_raw({"facet_candidates": [], "rationale": ""}))
    assert "Location" not in brief.permanent_filters


def test_v2_loader_string_geography_passes_through():
    from shared.brief_loader import _load_v2_brief

    brief = _load_v2_brief(_codex_geo_raw("New York City Metropolitan Area"))
    assert brief.permanent_filters["Location"] == "New York City Metropolitan Area"


def test_v2_loader_absent_geography_sets_no_location():
    from shared.brief_loader import _load_v2_brief

    brief = _load_v2_brief(_codex_geo_raw(None))
    assert "Location" not in brief.permanent_filters


def test_v2_loader_lane_hint_string_patterns_become_single_pattern():
    """patterns: "stripe" must become ["stripe"], never list("stripe") ==
    ["s","t","r","i","p","e"] — one-char patterns make the lane swallow
    every string in infer_domain_lane's substring matching."""
    from shared.brief_loader import _load_v2_brief

    raw = _codex_geo_raw(None)
    raw["domain_lane_hints"] = [{"lane": "payments_processors", "patterns": "stripe"}]
    brief = _load_v2_brief(raw)
    assert brief.domain_lane_hints[0].patterns == ["stripe"]


def test_candidate_register_terms_round_trip_to_new_and_compat_mirrors(
    tmp_path: Path,
) -> None:
    """candidate_register_terms are the search channel; key_terms remain eval."""

    from shared.brief_v2_schema import validate_v2_brief

    payload = _minimal_v2_brief()
    payload["capability_areas"][0]["key_terms"] = ["org design", "team topology"]
    payload["capability_areas"][0]["candidate_register_terms"] = [
        "engineering leadership",
        "scaled engineering teams",
    ]

    validate_v2_brief(payload)
    brief_path = _write_brief(tmp_path, payload)
    brief = load_brief(brief_path)

    area = brief._new_brief.capability_areas[0]
    assert area.key_terms == ["org design", "team topology"]
    assert area.candidate_register_terms == [
        "engineering leadership",
        "scaled engineering teams",
    ]
    assert brief.key_terms_by_area == {
        "Org leadership": ["org design", "team topology"]
    }
    assert brief.candidate_register_terms_by_area == {
        "Org leadership": ["engineering leadership", "scaled engineering teams"]
    }
