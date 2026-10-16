"""The ghost cursor revived through a rebrowser module alias (CLO-322).

``python_ghost_cursor.playwright_async`` imports type names from
``playwright.async_api``; the venv ships ``rebrowser_playwright`` and no
``playwright``, so the import has failed on every live run since the driver
switch (f857429). With ``LINKEDIN_GHOST_CURSOR_ENABLED`` on, the backend aliases
the two ``playwright`` module names to rebrowser before the import. Every test
here fakes the packages through ``sys.modules`` and the fixture below removes
any alias or ghost-cursor module a test leaves behind, so nothing leaks into
the rest of the suite.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin import input_backends
from linkedin.browser import LinkedInBrowser
from linkedin.input_backends import ConcurrentInputBackend
from linkedin.posture_report import describe_posture, format_posture
from shared import config
from shared.governor import UNGOVERNED_FOR_TESTS

_ALIAS_KEYS = ("playwright", "playwright.async_api")


@pytest.fixture(autouse=True)
def _no_alias_leak():
    saved = {key: sys.modules[key] for key in _ALIAS_KEYS if key in sys.modules}
    before = set(sys.modules)
    yield
    for key in _ALIAS_KEYS:
        sys.modules.pop(key, None)
    sys.modules.update(saved)
    for key in set(sys.modules) - before:
        if key.split(".")[0] in {"playwright", "python_ghost_cursor"}:
            sys.modules.pop(key, None)


def _flag(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(config, "LINKEDIN_GHOST_CURSOR_ENABLED", value, raising=False)


def _fake_find_spec(monkeypatch, *, playwright_installed: bool) -> list[str]:
    """Answer find_spec("playwright") as the test dictates; delegate the rest."""
    asked: list[str] = []
    real_find_spec = importlib.util.find_spec

    def fake(name, package=None):
        if name == "playwright":
            asked.append(name)
            return object() if playwright_installed else None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    return asked


def _fake_rebrowser(monkeypatch) -> tuple[types.ModuleType, types.ModuleType]:
    package = types.ModuleType("rebrowser_playwright")
    package.__path__ = []
    async_api = types.ModuleType("rebrowser_playwright.async_api")
    package.async_api = async_api
    monkeypatch.setitem(sys.modules, "rebrowser_playwright", package)
    monkeypatch.setitem(sys.modules, "rebrowser_playwright.async_api", async_api)
    return package, async_api


def _fake_ghost_cursor(monkeypatch, *, raises: Exception | None = None) -> list[object]:
    pages: list[object] = []

    def create_cursor(page):
        if raises is not None:
            raise raises
        pages.append(page)
        return types.SimpleNamespace(page=page)

    package = types.ModuleType("python_ghost_cursor")
    package.__path__ = []
    module = types.ModuleType("python_ghost_cursor.playwright_async")
    module.create_cursor = create_cursor
    package.playwright_async = module
    monkeypatch.setitem(sys.modules, "python_ghost_cursor", package)
    monkeypatch.setitem(sys.modules, "python_ghost_cursor.playwright_async", module)
    return pages


def _initialize(page=None) -> ConcurrentInputBackend:
    backend = ConcurrentInputBackend()
    asyncio.run(backend.initialize(page if page is not None else MagicMock()))
    return backend


def test_flag_on_without_playwright_aliases_rebrowser_and_creates_the_cursor(monkeypatch):
    _flag(monkeypatch, True)
    _fake_find_spec(monkeypatch, playwright_installed=False)
    package, async_api = _fake_rebrowser(monkeypatch)
    pages = _fake_ghost_cursor(monkeypatch)
    assert "playwright" not in sys.modules
    page = MagicMock()

    backend = _initialize(page)

    assert sys.modules["playwright"] is package
    assert sys.modules["playwright.async_api"] is async_api
    assert pages == [page]
    assert backend._cursor is not None
    assert backend.status_label == "ghost-cursor active (rebrowser alias)"


def test_flag_on_alias_survives_a_page_rebind(monkeypatch):
    # initialize runs again on every page (re)bind; once the alias is in
    # sys.modules, find_spec("playwright") answers with rebrowser's spec, and
    # the label must still say the cursor runs through the alias.
    _flag(monkeypatch, True)
    real_find_spec = importlib.util.find_spec

    def find_spec_as_the_import_system_would(name, package=None):
        if name == "playwright":
            return object() if name in sys.modules else None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_as_the_import_system_would)
    package, _ = _fake_rebrowser(monkeypatch)
    _fake_ghost_cursor(monkeypatch)

    backend = ConcurrentInputBackend()
    asyncio.run(backend.initialize(MagicMock()))
    asyncio.run(backend.initialize(MagicMock()))

    assert sys.modules["playwright"] is package
    assert backend.status_label == "ghost-cursor active (rebrowser alias)"


def test_flag_on_with_real_playwright_installs_no_alias(monkeypatch):
    _flag(monkeypatch, True)
    asked = _fake_find_spec(monkeypatch, playwright_installed=True)
    _fake_rebrowser(monkeypatch)
    pages = _fake_ghost_cursor(monkeypatch)
    page = MagicMock()

    backend = _initialize(page)

    assert asked == ["playwright"], "the flag-on path must look for a real playwright"
    for key in _ALIAS_KEYS:
        assert key not in sys.modules
    assert pages == [page]
    assert backend.status_label == "ghost-cursor active"


def test_flag_off_never_aliases_and_keeps_todays_failure(monkeypatch):
    _flag(monkeypatch, False)
    asked = _fake_find_spec(monkeypatch, playwright_installed=False)
    _fake_rebrowser(monkeypatch)
    helper = MagicMock(return_value=True)
    monkeypatch.setattr(input_backends, "_alias_playwright_to_rebrowser", helper, raising=False)
    # The import fails exactly as it does in the live venv today.
    monkeypatch.setitem(sys.modules, "python_ghost_cursor.playwright_async", None)

    backend = _initialize()

    helper.assert_not_called()
    assert asked == []
    for key in _ALIAS_KEYS:
        assert key not in sys.modules
    assert backend._cursor is None
    assert backend.status_label.startswith("ghost-cursor unavailable: ")


def test_flag_on_create_cursor_failure_stays_fail_soft(monkeypatch):
    _flag(monkeypatch, True)
    _fake_find_spec(monkeypatch, playwright_installed=False)
    _fake_rebrowser(monkeypatch)
    _fake_ghost_cursor(monkeypatch, raises=RuntimeError("cursor construction exploded"))

    backend = _initialize()

    assert backend._cursor is None
    assert backend.status_label == "ghost-cursor unavailable: cursor construction exploded"


def test_flag_on_alias_failure_stays_fail_soft(monkeypatch):
    _flag(monkeypatch, True)
    _fake_ghost_cursor(monkeypatch)

    def _boom():
        raise RuntimeError("alias exploded")

    monkeypatch.setattr(input_backends, "_alias_playwright_to_rebrowser", _boom, raising=False)

    backend = _initialize()

    assert backend._cursor is None
    assert backend.status_label == "ghost-cursor unavailable: alias exploded"


def test_posture_row_reports_the_alias(monkeypatch):
    _flag(monkeypatch, True)
    _fake_find_spec(monkeypatch, playwright_installed=False)
    _fake_rebrowser(monkeypatch)
    _fake_ghost_cursor(monkeypatch)

    rows = describe_posture(input_mode="concurrent")

    ghost_row = next(row for row in rows if row[0] == "ghost cursor")
    assert ghost_row == ("ghost cursor", True, "active (rebrowser alias)")
    assert format_posture([ghost_row]) == ["[ok] ghost cursor: active (rebrowser alias)"]


def test_posture_row_flag_off_keeps_the_missing_wording(monkeypatch):
    _flag(monkeypatch, False)
    asked = _fake_find_spec(monkeypatch, playwright_installed=False)
    _fake_rebrowser(monkeypatch)
    monkeypatch.setitem(sys.modules, "python_ghost_cursor.playwright_async", None)

    rows = describe_posture(input_mode="concurrent")

    ghost_row = next(row for row in rows if row[0] == "ghost cursor")
    assert ghost_row[1] is False
    assert asked == []
    assert "playwright" not in sys.modules
    assert format_posture([ghost_row])[0].startswith("[MISSING] ghost cursor: ")


def _require_the_live_venv_shape() -> None:
    pytest.importorskip("rebrowser_playwright.async_api")
    pytest.importorskip("python_ghost_cursor")
    if "playwright" in sys.modules or importlib.util.find_spec("playwright") is not None:
        pytest.skip("a real playwright is installed; the alias path is not exercised")


def test_live_venv_flag_off_reproduces_the_logged_failure(monkeypatch):
    _require_the_live_venv_shape()
    _flag(monkeypatch, False)

    backend = _initialize()

    assert backend._cursor is None
    assert backend.status_label == "ghost-cursor unavailable: No module named 'playwright'"


def test_live_venv_flag_on_imports_the_real_ghost_cursor(monkeypatch):
    _require_the_live_venv_shape()
    _flag(monkeypatch, True)
    import rebrowser_playwright

    page = MagicMock()
    backend = _initialize(page)

    assert sys.modules["playwright"] is rebrowser_playwright
    assert backend.status_label == "ghost-cursor active (rebrowser alias)"
    assert type(backend._cursor).__name__ == "GhostCursor"
    assert backend._cursor.page is page


_BOX = {"x": 100, "y": 200, "width": 40, "height": 20}


def _recording_page(events: list[tuple]) -> MagicMock:
    """The page surface the real GhostCursor.move and the backend commit touch."""
    page = MagicMock()
    page.mouse.move = AsyncMock(side_effect=lambda x, y, **_: events.append(("move", x, y)))
    page.mouse.down = AsyncMock(side_effect=lambda *a, **k: events.append(("down",)))
    page.mouse.up = AsyncMock(side_effect=lambda *a, **k: events.append(("up",)))
    page.context.new_cdp_session = AsyncMock(return_value=MagicMock())
    return page


def _locator_over(box) -> MagicMock:
    # GhostCursor.move(handle) calls handle.scroll_into_view_if_needed() and
    # handle.bounding_box(); nothing else on the handle.
    handle = MagicMock()
    handle.scroll_into_view_if_needed = AsyncMock()
    handle.bounding_box = AsyncMock(return_value=box)
    locator = MagicMock()
    locator.scroll_into_view_if_needed = AsyncMock()
    locator.element_handle = AsyncMock(return_value=handle)
    locator.hover = AsyncMock()
    return locator


def _ghost_click_through_the_browser(page, locator) -> bool:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = page

    async def run():
        await browser._input_backend.initialize(page)
        return await browser._ghost_click_locator(locator)

    # The backend's post-click pause is up to two real seconds.
    with patch("linkedin.input_backends.asyncio.sleep", new=AsyncMock()):
        return asyncio.run(run())


def test_live_venv_ghost_click_traces_a_path_through_the_real_cursor(monkeypatch):
    _require_the_live_venv_shape()
    _flag(monkeypatch, True)
    events: list[tuple] = []
    page = _recording_page(events)
    locator = _locator_over(dict(_BOX))

    clicked = _ghost_click_through_the_browser(page, locator)

    moves = [event for event in events if event[0] == "move"]
    assert clicked is True
    assert len(moves) > 1, f"expected a traced path, got {len(moves)} move(s)"
    _, last_x, last_y = moves[-1]
    assert _BOX["x"] <= last_x <= _BOX["x"] + _BOX["width"]
    assert _BOX["y"] <= last_y <= _BOX["y"] + _BOX["height"]
    assert events[len(moves):] == [("down",), ("up",)]
    locator.hover.assert_not_awaited()


def test_live_venv_detached_element_falls_back_to_hover(monkeypatch):
    _require_the_live_venv_shape()
    _flag(monkeypatch, True)
    events: list[tuple] = []
    page = _recording_page(events)
    locator = _locator_over(None)

    clicked = _ghost_click_through_the_browser(page, locator)

    assert clicked is True
    locator.hover.assert_awaited_once()
    assert events == [("down",), ("up",)]
