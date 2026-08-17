"""MAX_PAGES_PER_STRING config defaults and consumer semantics."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from shared.schemas import Progress, SearchString


def _fresh_config_max_pages(monkeypatch, raw_value: str | None) -> int:
    import shared

    old_module = sys.modules.pop("shared.config", None)
    old_attr = getattr(shared, "config", None)
    try:
        with monkeypatch.context() as env:
            if raw_value is None:
                env.delenv("MAX_PAGES_PER_STRING", raising=False)
            else:
                env.setenv("MAX_PAGES_PER_STRING", raw_value)
            config = importlib.import_module("shared.config")
            return config.MAX_PAGES_PER_STRING
    finally:
        sys.modules.pop("shared.config", None)
        if old_module is not None:
            sys.modules["shared.config"] = old_module
            setattr(shared, "config", old_module)
        elif old_attr is not None:
            setattr(shared, "config", old_attr)
        elif hasattr(shared, "config"):
            delattr(shared, "config")


def _make_pipeline(output_dir: str):
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.permanent_filters = {}
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        return Pipeline(brief_path=str(brief_path), output_dir=output_dir)


def test_max_pages_per_string_default_is_25(monkeypatch):
    assert _fresh_config_max_pages(monkeypatch, None) == 25


def test_max_pages_per_string_explicit_40_passes_through(monkeypatch):
    assert _fresh_config_max_pages(monkeypatch, "40") == 40


def test_max_pages_per_string_explicit_zero_is_unbounded_at_consumer(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=11, name="sub-500 pool", boolean="foo", status="queued")
    progress = Progress(
        brief_name="test",
        strings=[search_string],
        current_string_id=11,
        current_page=0,
    )

    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    locator = MagicMock()
    locator.first = no_results

    pipeline.browser.page = MagicMock(url="https://www.linkedin.com/talent/search")
    pipeline.browser.page.locator.return_value = locator
    pipeline.browser.get_results_count_text = AsyncMock(return_value="386")
    pipeline.browser.get_results_count = AsyncMock(return_value=386)
    pipeline.browser.go_to_next_page = AsyncMock(side_effect=[True] * 25 + [False])
    pipeline._apply_opening_search = AsyncMock()
    pipeline._ensure_browser_healthy = AsyncMock()
    pipeline._review_page_sequentially = AsyncMock(return_value=None)
    pipeline._assess_string_state = AsyncMock(
        return_value={
            "decision": "continue",
            "rationale": "test keeps paginating",
            "page_signal": False,
            "committed_zero_signal_streak": 0,
        }
    )
    pipeline._evaluate_variant_lifecycle = MagicMock(return_value=None)
    pipeline._checkpoint_progress = MagicMock()

    with patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 0), \
         patch("linkedin.orchestrator.asyncio.sleep", new=AsyncMock()):
        asyncio.run(pipeline._process_string(search_string, progress))

    assert pipeline._review_page_sequentially.await_count == 26
    assert pipeline.browser.go_to_next_page.await_count == 26
