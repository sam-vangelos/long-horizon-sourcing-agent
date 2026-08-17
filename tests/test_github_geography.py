"""Bay Area geography enforcement for the GitHub pipeline."""

from __future__ import annotations

from github.schemas import GitHubCandidate

from tests.test_github_pipeline import (
    GitHubPipeline,
    _make_candidate,
    _make_pipeline,
    _make_query,
)


class _NewBrief:
    def __init__(self, geography: str = ""):
        self.geography = geography


def _pipeline_with_bay_area_geo() -> GitHubPipeline:
    pipeline = _make_pipeline()
    pipeline.brief_obj.has_v2_schema = True
    pipeline.brief_obj._new_brief = _NewBrief(geography="San Francisco Bay Area")
    pipeline.brief_obj.permanent_filters = {}
    return pipeline


def _candidate_with_location(location: str) -> GitHubCandidate:
    candidate = _make_candidate("bay_user", "Bay User")
    candidate.user.location = location
    return candidate


def test_stated_san_francisco_passes() -> None:
    pipeline = _pipeline_with_bay_area_geo()
    query = _make_query(channel="code_search")

    assert pipeline._passes_geography_check(
        _candidate_with_location("San Francisco"), query
    ) is True
    assert pipeline.stats.get("geo_rejected_stated", 0) == 0


def test_stated_beijing_rejects() -> None:
    pipeline = _pipeline_with_bay_area_geo()
    query = _make_query(channel="code_search")

    assert pipeline._passes_geography_check(
        _candidate_with_location("Beijing"), query
    ) is False
    assert pipeline.stats["geo_rejected_stated"] == 1


def test_los_angeles_ca_rejects_despite_california_substring() -> None:
    pipeline = _pipeline_with_bay_area_geo()
    query = _make_query(channel="code_search")

    assert pipeline._passes_geography_check(
        _candidate_with_location("Los Angeles, CA"), query
    ) is False
    assert pipeline.stats["geo_rejected_stated"] == 1


def test_stated_london_rejects() -> None:
    pipeline = _pipeline_with_bay_area_geo()
    query = _make_query(channel="code_search")

    assert pipeline._passes_geography_check(
        _candidate_with_location("London, UK"), query
    ) is False
    assert pipeline.stats["geo_rejected_stated"] == 1


def test_blank_location_passes_with_unverified_marker() -> None:
    pipeline = _pipeline_with_bay_area_geo()
    candidate = _candidate_with_location("")
    query = _make_query(channel="code_search")

    assert pipeline._passes_geography_check(candidate, query) is True
    assert candidate.portfolio_summary["_geo_status"] == "unverified"
    assert pipeline.stats.get("geo_rejected_stated", 0) == 0


def test_no_geography_brief_passes_everyone() -> None:
    pipeline = _make_pipeline()
    pipeline.brief_obj.has_v2_schema = True
    pipeline.brief_obj._new_brief = _NewBrief(geography="")
    pipeline.brief_obj.permanent_filters = {}
    query = _make_query(channel="code_search")

    for location in ("Beijing", "San Francisco", "Los Angeles, CA", ""):
        candidate = _candidate_with_location(location)
        assert pipeline._passes_geography_check(candidate, query) is True

    assert pipeline.stats.get("geo_rejected_stated", 0) == 0
