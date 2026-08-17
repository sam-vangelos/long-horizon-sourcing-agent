"""Tests for decoy tab ownership and cleanup behavior."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from decoy.agent import DecoyAgent


def test_ensure_tab_creates_dedicated_page_instead_of_reusing_existing_linkedin_tab():
    existing_page = MagicMock()
    existing_page.is_closed.return_value = False

    decoy_page = MagicMock()
    decoy_page.goto = AsyncMock()
    decoy_page.is_closed.return_value = False

    context = MagicMock()
    context.pages = [existing_page]
    context.new_page = AsyncMock(return_value=decoy_page)

    agent = DecoyAgent(context)
    agent._init_cursor = MagicMock()

    with patch("decoy.agent.asyncio.sleep", new=AsyncMock()):
        asyncio.run(agent._ensure_tab())

    context.new_page.assert_awaited_once()
    decoy_page.goto.assert_awaited_once()
    agent._init_cursor.assert_called_once()
    assert agent._page is decoy_page
    assert agent._owns_page is True


def test_close_shuts_owned_decoy_page():
    page = MagicMock()
    page.is_closed.return_value = False
    page.goto = AsyncMock()
    page.close = AsyncMock()

    agent = DecoyAgent(MagicMock())
    agent._page = page
    agent._owns_page = True
    agent._cursor = object()

    asyncio.run(agent.close())

    page.goto.assert_awaited_once()
    page.close.assert_awaited_once()
    assert agent._page is None
    assert agent._cursor is None
    assert agent._owns_page is False


def test_close_leaves_unowned_page_open():
    page = MagicMock()
    page.is_closed.return_value = False
    page.goto = AsyncMock()
    page.close = AsyncMock()

    agent = DecoyAgent(MagicMock())
    agent._page = page
    agent._owns_page = False

    asyncio.run(agent.close())

    page.goto.assert_awaited_once()
    page.close.assert_not_awaited()
    assert agent._page is None
    assert agent._owns_page is False
