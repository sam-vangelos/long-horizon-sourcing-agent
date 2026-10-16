"""Input-backend routing lock for the legacy Location filter (CLO-163).

``_set_location_filter`` is the one control that predated the input-backend
seam: it clicked and filled the raw locator, so under ``--input-mode away``
(OS-level HID) that single interaction silently reverted to synthetic CDP
input. This locks the route: clicks via the backend wrapper, text via
``InputBackend.type_text``, and never ``locator.click()`` / ``locator.fill()``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from linkedin.browser import LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS


def _fake_locator() -> MagicMock:
    locator = MagicMock()
    locator.wait_for = AsyncMock()
    locator.click = AsyncMock()
    locator.fill = AsyncMock()
    locator.scroll_into_view_if_needed = AsyncMock()
    locator.hover = AsyncMock()
    return locator


def test_set_location_filter_routes_through_input_backend():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    loc_input = _fake_locator()
    option = _fake_locator()

    def _locator(selector: str) -> MagicMock:
        holder = MagicMock()
        holder.first = loc_input if "input[" in selector else option
        return holder

    page = MagicMock()
    page.locator = MagicMock(side_effect=_locator)
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    browser._input_backend.click_locator = AsyncMock(return_value=True)
    browser._input_backend.type_text = AsyncMock()

    asyncio.run(browser._set_location_filter("Berlin"))

    # The raw-locator paths must be dead: no synthetic click/fill anywhere.
    loc_input.click.assert_not_awaited()
    loc_input.fill.assert_not_awaited()
    option.click.assert_not_awaited()
    # Text entry goes through the backend with a human typing plan attached.
    browser._input_backend.type_text.assert_awaited_once()
    assert browser._input_backend.type_text.await_args.args[2] == "Berlin"
    assert browser._input_backend.type_text.await_args.kwargs["plan"] is not None
    # Both clicks (input focus + typeahead option) route through the backend.
    assert browser._input_backend.click_locator.await_count == 2
