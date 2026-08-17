"""Generic browser/session helper for identity-resolution retrieval experiments."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from shared import config

if TYPE_CHECKING:
    from rebrowser_playwright.async_api import Browser, BrowserContext, Page


class IdentityExperimentBrowser:
    """CDP-attached browser helper for Bing and public LinkedIn experiment flows."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser: Browser | None = None
        self._shared_context: BrowserContext | None = None
        self._linkedin_context: BrowserContext | None = None
        self._surface_pages: dict[str, Page] = {}

    async def connect(self) -> None:
        if self._browser is not None:
            return
        from rebrowser_playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.connect_over_cdp(config.CDP_URL)
        if not self._browser.contexts:
            raise RuntimeError("No browser contexts found. Is the browser open?")

        self._shared_context = self._browser.contexts[0]
        self._linkedin_context = self._select_linkedin_context() or self._shared_context

    async def disconnect(self) -> None:
        for page in list(self._surface_pages.values()):
            try:
                await page.close()
            except Exception:
                pass
        self._surface_pages = {}
        if self._playwright is not None:
            await self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._shared_context = None
        self._linkedin_context = None

    def _select_linkedin_context(self) -> BrowserContext | None:
        if not self._browser:
            return None
        for context in self._browser.contexts:
            for page in context.pages:
                try:
                    url = page.url
                except Exception:
                    continue
                if "linkedin.com" in url:
                    return context
        return None

    @asynccontextmanager
    async def strategy_page(self, surface: str):
        if self._browser is None:
            raise RuntimeError("Experiment browser is not connected")

        context = self._context_for_surface(surface)
        if context is None:
            raise RuntimeError("No browser context available for experiment strategy")

        page = self._surface_pages.get(surface)
        if page is None or page.is_closed():
            page = await context.new_page()
            self._surface_pages[surface] = page
        yield page

    def _context_for_surface(self, surface: str) -> BrowserContext | None:
        if surface == "web":
            return self._shared_context
        if surface in {"linkedin", "recruiter"}:
            return self._linkedin_context or self._shared_context
        return self._shared_context

    async def detect_blocker_state(self, page: "Page", *, surface: str) -> str:
        url = ""
        try:
            url = page.url
        except Exception:
            url = ""

        if "captcha" in url.lower() or "challenge" in url.lower():
            return "captcha"
        if surface in {"linkedin", "recruiter"}:
            if "/login" in url or "session_key" in url:
                return "login_wall"
            try:
                if await page.locator('input[name="session_key"]').first.is_visible(timeout=500):
                    return "login_wall"
            except Exception:
                pass
        try:
            body = (await page.locator("body").inner_text(timeout=1000)).lower()
        except Exception:
            body = ""
        if "captcha" in body or "verify you are human" in body:
            return "captcha"
        if surface in {"linkedin", "recruiter"} and any(
            phrase in body
            for phrase in (
                "sign in",
                "join now",
                "please sign in",
            )
        ):
            return "login_wall"
        return ""
