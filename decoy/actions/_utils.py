"""Shared utilities for decoy actions — ghost-cursor clicks and human scrolling."""

import asyncio
import math
import random
from shared.human_timing import human_delay, human_delay_correlated


async def ghost_click(cursor, page, selector_or_locator):
    """Click using ghost-cursor Bézier trajectory, fallback to Playwright .click().

    Mirrors LinkedInBrowser._ghost_click() from browser.py — same timeout,
    same fallback pattern, so both agents produce identical click signatures.

    Args:
        cursor: python_ghost_cursor cursor instance (or None)
        page: Playwright page object
        selector_or_locator: CSS selector string, Playwright Locator, or ElementHandle
    """
    if cursor:
        try:
            # ghost-cursor accepts CSS selectors and ElementHandles
            await asyncio.wait_for(cursor.click(selector_or_locator), timeout=5.0)
            return
        except Exception:
            pass

    # Fallback: use Playwright click
    if isinstance(selector_or_locator, str):
        locator = page.locator(selector_or_locator).first
        await locator.click(timeout=5000)
    else:
        # ElementHandle or Locator
        await selector_or_locator.click(timeout=5000)


async def human_scroll(page, delta_y, *, channel: str = "scroll"):
    """Scroll using chunked page.mouse.wheel() calls for realistic behavior.

    Uses page.mouse.wheel() which dispatches isTrusted:true mouseWheel events
    via Playwright's internal CDP channel — no manual CDP session management.

    Args:
        page: Playwright page object
        delta_y: Total scroll distance in pixels (positive = down)
        channel: timing stream name for correlated pacing
    """
    if delta_y == 0:
        return 0

    direction = 1 if delta_y > 0 else -1
    remaining = abs(delta_y)
    wheel_events = 0

    while remaining > 0:
        base_chunk = int(random.lognormvariate(math.log(85), 0.35))
        if remaining < 120:
            base_chunk = min(base_chunk, random.randint(24, 80))
        chunk = min(remaining, max(20, base_chunk))
        await page.mouse.wheel(0, chunk * direction)
        wheel_events += 1
        remaining -= chunk
        await asyncio.sleep(
            human_delay_correlated(0.03, spread=0.45, channel=f"{channel}_micro")
        )

        if remaining > 0 and random.random() < 0.10:
            await asyncio.sleep(
                human_delay_correlated(
                    random.uniform(0.12, 0.35),
                    spread=0.5,
                    channel=f"{channel}_pause",
                )
            )

    # A small fraction of scrolls include a corrective nudge so the wheel
    # trace is not perfectly monotonic.
    if abs(delta_y) >= 220 and random.random() < 0.18:
        correction = min(random.randint(12, 36), max(12, abs(delta_y) // 8))
        await page.mouse.wheel(0, -correction * direction)
        wheel_events += 1
        await asyncio.sleep(
            human_delay_correlated(0.04, spread=0.4, channel=f"{channel}_micro")
        )
        if random.random() < 0.75:
            await page.mouse.wheel(0, correction * direction)
            wheel_events += 1
    return wheel_events
