"""Adversarial coverage for `non_fit_patterns` coercion in `_load_v2_brief`.

The implementer added `_normalize_non_fit_patterns` to tolerate the
string-shaped `non_fit_patterns` the conversational extractor emits. This
file pressure-tests the corners that helper still has to survive:

- a MIXED list of strings AND dicts,
- empty / whitespace-only strings (must be dropped, not coerced),
- a dict element missing `label` (does the `NonFitPattern` build crash?),
- a dict element missing `why_not` (the second mandatory subscript),
- a non-str/non-dict element (int, None, nested list) interleaved,
- a very long string against the `[:80]` label slice,
- the SECOND subscript site (`noise_archetypes`) under the same shapes.

These exercise `load_brief` end-to-end (write JSON, load, assert) exactly
like the implementer's test, so a regression at either subscript site
surfaces as a `TypeError` rather than a silent shape change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.brief_loader import load_brief


def _minimal_v2_brief() -> dict:
    return {
        "role_title": "VP Engineering",
        "role_summary": "Owns engineering org for a series-C company.",
        "geography": "United States",
        "linkedin_project": "exec-search-vp-eng",
        "minimum_years_experience": 12,
        "minimum_bar_description": "10+ years engineering leadership.",
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


def _load(tmp_path: Path, non_fit_patterns: object):
    payload = _minimal_v2_brief()
    payload["non_fit_patterns"] = non_fit_patterns
    return load_brief(_write_brief(tmp_path, payload))


# ---------------------------------------------------------------------------
# Mixed list: strings AND dicts coexist (extractor output later edited by the
# composer, or a half-migrated draft). Both must survive and keep order.
# ---------------------------------------------------------------------------


def test_mixed_string_and_dict_non_fit_patterns(tmp_path: Path) -> None:
    brief = _load(
        tmp_path,
        [
            "Title without ownership",
            {
                "label": "AI adjacency only",
                "why_not": "Mentions AI in profile but never shipped it.",
                "description": "Buzzword adjacency, no delivery.",
                "examples": ["prompt tinkering"],
            },
        ],
    )
    nfp = brief._new_brief.non_fit_patterns
    assert len(nfp) == 2
    # String element coerced.
    assert nfp[0].label == "Title without ownership"
    assert nfp[0].why_not == "Title without ownership"
    # Dict element passes through with its own fields intact.
    assert nfp[1].label == "AI adjacency only"
    assert nfp[1].why_not == "Mentions AI in profile but never shipped it."
    assert nfp[1].description == "Buzzword adjacency, no delivery."
    assert nfp[1].examples == ["prompt tinkering"]
    # Second subscript site agrees.
    assert [a["name"] for a in brief.noise_archetypes] == [
        "Title without ownership",
        "AI adjacency only",
    ]


# ---------------------------------------------------------------------------
# Empty / whitespace strings must be DROPPED, not coerced into empty-label
# NonFitPatterns. The composer mirror drops them; the loader should too.
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_strings_dropped(tmp_path: Path) -> None:
    brief = _load(tmp_path, ["", "   ", "\t\n", "Real pattern"])
    nfp = brief._new_brief.non_fit_patterns
    assert len(nfp) == 1
    assert nfp[0].label == "Real pattern"
    assert len(brief.noise_archetypes) == 1


# ---------------------------------------------------------------------------
# Non-str/non-dict junk interleaved (int, None, nested list). These should be
# silently dropped, not crash and not appear as patterns.
# ---------------------------------------------------------------------------


def test_non_str_non_dict_elements_dropped(tmp_path: Path) -> None:
    brief = _load(tmp_path, [None, 42, ["nested"], "Kept", {"label": "K2", "why_not": "w"}])
    nfp = brief._new_brief.non_fit_patterns
    labels = [p.label for p in nfp]
    assert labels == ["Kept", "K2"]


# ---------------------------------------------------------------------------
# Long string vs the [:80] label slice: label truncates, why_not keeps full.
# ---------------------------------------------------------------------------


def test_long_string_label_truncated_whynot_full(tmp_path: Path) -> None:
    long = "A" * 200
    brief = _load(tmp_path, [long])
    nfp = brief._new_brief.non_fit_patterns
    assert len(nfp) == 1
    assert nfp[0].label == "A" * 80
    assert nfp[0].why_not == "A" * 200
    # noise_archetypes mirror takes label from the same normalized dict.
    assert brief.noise_archetypes[0]["name"] == "A" * 80


# ---------------------------------------------------------------------------
# HOLE CANDIDATE: a dict element MISSING `label`. The implementer's helper
# passes any dict through unchanged (unlike composer._normalize_patterns,
# which requires label AND why_not). The NonFitPattern build then does
# `nf["label"]` -> KeyError if the helper does not synthesize a label.
# ---------------------------------------------------------------------------


def test_dict_missing_label_does_not_crash(tmp_path: Path) -> None:
    brief = _load(tmp_path, [{"why_not": "Never owned the roadmap."}])
    # We don't assert a specific shape here — only that loading a dict with no
    # `label` does NOT raise. If the loader chooses to drop it, len==0 is fine;
    # if it synthesizes a label, len==1 is fine. A crash is the failure.
    assert brief._new_brief is not None


# ---------------------------------------------------------------------------
# HOLE CANDIDATE: a dict element MISSING `why_not`. NonFitPattern build does
# `why_not=nf["why_not"]` -> KeyError. composer._normalize_patterns drops such
# dicts; the loader helper does not.
# ---------------------------------------------------------------------------


def test_dict_missing_why_not_does_not_crash(tmp_path: Path) -> None:
    brief = _load(tmp_path, [{"label": "Has label, no why_not"}])
    assert brief._new_brief is not None


# ---------------------------------------------------------------------------
# Regression guard: the composer's canonical dict shape (label + why_not +
# description + examples) must still hydrate unchanged through the new path.
# ---------------------------------------------------------------------------


def test_canonical_dict_shape_still_loads(tmp_path: Path) -> None:
    brief = _load(
        tmp_path,
        [
            {
                "label": "Title without ownership",
                "description": "Holds a senior title but never owned delivery.",
                "why_not": "No evidence of org-level ownership.",
                "examples": ["Director with no reports"],
            }
        ],
    )
    nfp = brief._new_brief.non_fit_patterns
    assert len(nfp) == 1
    assert nfp[0].label == "Title without ownership"
    assert nfp[0].description == "Holds a senior title but never owned delivery."
    assert nfp[0].why_not == "No evidence of org-level ownership."
    assert nfp[0].examples == ["Director with no reports"]
    arch = brief.noise_archetypes[0]
    assert arch["name"] == "Title without ownership"
    assert arch["description"] == "Holds a senior title but never owned delivery."
    assert arch["signals"] == ["Director with no reports"]
