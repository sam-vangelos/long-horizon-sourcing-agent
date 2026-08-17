"""Unit tests for the extracted LinkedIn GeographyGateService cluster."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from shared.schemas import CandidateSnippet

from linkedin.geography_gate import GeographyGateDeps, GeographyGateService


def _make_snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Test Person",
        "headline": "",
        "current_title": "",
        "current_company": "",
        "location": "Somewhere",
        "education_snippet": "",
        "profile_url": "/talent/profile/test123",
        "source_string_id": 1,
        "source_string_name": "test",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


def _make_service(
    tmp_path: Path,
    *,
    location_filter: str = "San Francisco Bay Area",
) -> GeographyGateService:
    brief = MagicMock()
    brief.id = "test"
    brief.permanent_filters = {"Location": location_filter}
    brief.geography_source = "operator"
    browser = MagicMock()

    deps = GeographyGateDeps(
        get_browser=lambda: browser,
        get_brief_obj=lambda: brief,
        log_path=tmp_path / "run_log.jsonl",
        stats={"off_geo_saves": 0},
    )
    return GeographyGateService(deps)


def test_candidate_location_contained_in_geo_and_off_geo():
    """Containment verdicts match the service's cheap_llm-backed logic."""
    with tempfile.TemporaryDirectory() as td:
        service = _make_service(Path(td))
        geo_values = ["San Francisco Bay Area"]

        with patch("shared.llm_clients.cheap_llm") as mock_llm:
            mock_llm.return_value = {"contained": True}
            assert (
                service._candidate_location_contained("Mountain View, California", geo_values)
                is True
            )

            mock_llm.return_value = {"contained": False}
            assert (
                service._candidate_location_contained("São Paulo, Brazil", geo_values)
                is False
            )


def test_warn_if_off_geo_save_only_for_off_geo():
    """Off-geo saves increment stats and print; in-geo saves stay silent."""
    with tempfile.TemporaryDirectory() as td:
        service = _make_service(Path(td))

        with patch("shared.llm_clients.cheap_llm") as mock_llm:
            mock_llm.return_value = {"contained": True}
            in_geo = _make_snippet(location="Mountain View, California")
            with patch("builtins.print") as mock_print:
                service._warn_if_off_geo_save(in_geo)
            assert service.deps.stats["off_geo_saves"] == 0
            mock_print.assert_not_called()

            mock_llm.return_value = {"contained": False}
            off_geo = _make_snippet(location="São Paulo, Brazil")
            with patch("builtins.print") as mock_print:
                service._warn_if_off_geo_save(off_geo)
            assert service.deps.stats["off_geo_saves"] == 1
            mock_print.assert_called_once()
            assert "[geo-warn]" in mock_print.call_args[0][0]


def test_containment_cache_survives_across_calls():
    """The containment cache is populated on first use and reused."""
    with tempfile.TemporaryDirectory() as td:
        service = _make_service(Path(td))
        geo_values = ["San Francisco Bay Area"]
        location = "Mountain View, California"

        with patch("shared.llm_clients.cheap_llm") as mock_llm:
            mock_llm.return_value = {"contained": True}
            first = service._candidate_location_contained(location, geo_values)
            second = service._candidate_location_contained(location, geo_values)

        assert first is True
        assert second is True
        mock_llm.assert_called_once()
        assert hasattr(service, "_geo_containment_cache")
        assert (location.lower(), tuple(geo_values)) in service._geo_containment_cache


def test_service_reads_browser_live_not_snapshotted():
    """get_browser must read live state, not a browser snapshotted at construction.

    The staleness mode this locks is REBINDING, not in-place mutation. A test
    that swaps the pipeline's browser after init rebinds to a NEW object; a
    snapshot field would keep pointing at the browser captured at construction.

    Note: in-place mutation of one mock proves nothing — a snapshotted VALUE
    field holds that same object, so mutation is visible under both designs.
    """
    with tempfile.TemporaryDirectory() as td:
        service = _make_service(Path(td))
        holder: dict = {"browser": MagicMock()}
        holder["browser"].apply_location_filter = AsyncMock(return_value=True)

        deps = replace(
            service.deps,
            get_browser=lambda: holder["browser"],
        )
        service = GeographyGateService(deps)

        asyncio.run(service._apply_session_location_filter())
        first_browser = holder["browser"]
        first_browser.apply_location_filter.assert_awaited_once()

        new_browser = MagicMock()
        new_browser.apply_location_filter = AsyncMock(return_value=True)
        holder["browser"] = new_browser

        service._session_location_applied = False
        asyncio.run(service._apply_session_location_filter())

        new_browser.apply_location_filter.assert_awaited_once()
        assert first_browser.apply_location_filter.await_count == 1
