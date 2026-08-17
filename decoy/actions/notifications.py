"""Notification checking — scan, occasionally click through one."""

import asyncio
import random
import time
from shared.human_timing import human_delay
from decoy.actions._utils import ghost_click


async def check_notifications(page, cursor=None) -> dict:
    """Click the notifications bell, scan briefly, optionally click one.

    Args:
        page: Playwright page object
        cursor: python_ghost_cursor instance for Bézier click trajectories

    Returns:
        dict with action metadata
    """
    start = time.time()
    clicked_through = False

    # Navigate to notifications via bell icon (ghost-cursor click)
    try:
        bell = page.locator("#global-nav a[href*='notifications']").first
        if await bell.count() > 0:
            await ghost_click(cursor, page, "#global-nav a[href*='notifications']")
        else:
            await page.goto("https://www.linkedin.com/notifications/", wait_until="domcontentloaded")
    except Exception:
        await page.goto("https://www.linkedin.com/notifications/", wait_until="domcontentloaded")

    await asyncio.sleep(human_delay(1.5, 3.0))

    # Scan for 5-15 seconds
    await asyncio.sleep(random.uniform(5, 15))

    # 20% chance to click into one notification
    if random.random() < 0.2:
        try:
            card_list = await page.locator(".nt-card, .notification-card").all()
            if card_list and len(card_list) > 0:
                idx = random.randint(0, min(len(card_list) - 1, 4))
                await ghost_click(cursor, page, card_list[idx])
                clicked_through = True
                # Dwell on notification detail
                await asyncio.sleep(random.uniform(3, 8))
                # Go back
                await page.goto("https://www.linkedin.com/notifications/", wait_until="domcontentloaded")
                await asyncio.sleep(human_delay(1.0, 2.0))
        except Exception:
            pass

    # Return to feed
    await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")

    duration = round(time.time() - start, 1)
    return {"type": "notification_check", "clicked_through": clicked_through, "duration": duration}
