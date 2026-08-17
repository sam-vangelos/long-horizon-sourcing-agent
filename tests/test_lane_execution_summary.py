"""Tests for lane execution summary in run snapshots (D1).

Pins:
- Lane summary groups correctly from mixed-lane strings.
- Old snapshots without lane_execution_summary load without error.
- string_performance is untouched by the addition.
- Research context extraction prefers the pre-built key when present.
"""

from __future__ import annotations

import pytest

from market_intelligence.research_context import _lane_execution_summary


def _sp(
    string_id: int,
    domain_lane: str = "",
    family_key: str = "",
    result_count: int = 100,
    pages_reviewed: int = 4,
    saves: int = 2,
    facial_yes_count: int = 10,
    facial_no_count: int = 80,
    candidates_count: int = 100,
) -> dict:
    return {
        "string_id": string_id,
        "name": f"string-{string_id}",
        "status": "done",
        "result_count": result_count,
        "pages_reviewed": pages_reviewed,
        "saves": saves,
        "save_rate": round(saves / max(candidates_count, 1), 4),
        "saved_candidates": [],
        "notes": "",
        "facial_yes_count": facial_yes_count,
        "facial_no_count": facial_no_count,
        "candidates_count": candidates_count,
        "duplicates_count": 0,
        "family_key": family_key,
        "novelty_bucket": "",
        "domain_lane": domain_lane,
    }


# -- Extraction prefers pre-built key --

def test_prefers_prebuilt_lane_execution_summary():
    prebuilt = [{"lane_id": "ml-infra", "saves": 5}]
    report_input = {"lane_execution_summary": prebuilt}
    result = _lane_execution_summary(report_input, [])
    assert result == prebuilt


def test_ignores_empty_prebuilt_key():
    report_input = {"lane_execution_summary": []}
    sp = [_sp(1, domain_lane="alpha")]
    result = _lane_execution_summary(report_input, sp)
    assert len(result) == 1
    assert result[0]["lane_id"] == "alpha"


# -- Reconstruction from string_performance --

def test_groups_by_domain_lane():
    sp = [
        _sp(1, domain_lane="ml-infra", saves=3, candidates_count=50),
        _sp(2, domain_lane="ml-infra", saves=1, candidates_count=50),
        _sp(3, domain_lane="platform", saves=2, candidates_count=100),
    ]
    result = _lane_execution_summary(None, sp)
    by_id = {r["lane_id"]: r for r in result}
    assert set(by_id.keys()) == {"ml-infra", "platform"}
    assert by_id["ml-infra"]["saves"] == 4
    assert by_id["ml-infra"]["candidates_evaluated"] == 100
    assert by_id["ml-infra"]["string_count"] == 2
    assert by_id["platform"]["saves"] == 2


def test_falls_back_to_family_key_when_no_domain_lane():
    sp = [
        _sp(1, family_key="infra-focus"),
        _sp(2, family_key="infra-focus"),
    ]
    result = _lane_execution_summary(None, sp)
    assert len(result) == 1
    assert result[0]["lane_id"] == "infra-focus"
    assert result[0]["string_count"] == 2


def test_legacy_bucket_for_strings_without_lane_data():
    sp = [_sp(1), _sp(2)]
    result = _lane_execution_summary(None, sp)
    assert len(result) == 1
    assert result[0]["lane_id"] == "legacy"


def test_mixed_lane_and_legacy_strings():
    sp = [
        _sp(1, domain_lane="alpha", saves=5, candidates_count=50),
        _sp(2, saves=1, candidates_count=50),
    ]
    result = _lane_execution_summary(None, sp)
    by_id = {r["lane_id"]: r for r in result}
    assert set(by_id.keys()) == {"alpha", "legacy"}
    assert by_id["alpha"]["saves"] == 5
    assert by_id["legacy"]["saves"] == 1


def test_save_rate_computed_per_lane():
    sp = [
        _sp(1, domain_lane="a", saves=10, candidates_count=100),
        _sp(2, domain_lane="a", saves=0, candidates_count=100),
    ]
    result = _lane_execution_summary(None, sp)
    assert result[0]["save_rate"] == round(10 / 200, 4)


def test_family_keys_deduplicated():
    sp = [
        _sp(1, domain_lane="a", family_key="fk1"),
        _sp(2, domain_lane="a", family_key="fk1"),
        _sp(3, domain_lane="a", family_key="fk2"),
    ]
    result = _lane_execution_summary(None, sp)
    assert sorted(result[0]["family_keys"]) == ["fk1", "fk2"]


def test_empty_string_performance():
    result = _lane_execution_summary(None, [])
    assert result == []


def test_old_snapshot_no_key_no_error():
    """Old snapshots without lane_execution_summary should reconstruct without error."""
    report_input = {
        "string_performance": [
            _sp(1, family_key="test"),
        ]
    }
    result = _lane_execution_summary(report_input, report_input["string_performance"])
    assert len(result) == 1
    assert result[0]["lane_id"] == "test"
