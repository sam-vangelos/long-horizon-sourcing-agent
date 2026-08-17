"""Feed scrolling — passive, no engagement."""

import asyncio
import json
import random
import time
from shared.human_timing import human_delay
from decoy.actions._utils import human_scroll


async def scroll_feed(page, cursor=None) -> dict:
    """Scroll the LinkedIn feed passively.

    No likes, comments, shares, or expansions. Variable scroll depth.
    Dwell on ~30-50% of posts as they scroll into view.

    Args:
        page: Playwright page object (on linkedin.com/feed)
        cursor: python_ghost_cursor instance (unused for feed — no clicks)

    Returns:
        dict with action metadata (type, duration)
    """
    start = time.time()

    # Navigate to feed if not already there
    if "/feed" not in page.url:
        await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
        await asyncio.sleep(human_delay(1.5, 3.0))

    vp = json.loads(await page.evaluate(
        "JSON.stringify({w: window.innerWidth, h: window.innerHeight})"
    ))
    vh = vp["h"]

    # Variable scroll depth: 3-10 screen-heights
    screen_heights = random.randint(3, 10)
    total_scroll = screen_heights * vh
    scrolled = 0

    while scrolled < total_scroll:
        # Scroll one partial screen-height
        scroll_amount = int(vh * random.uniform(0.3, 0.8))
        await human_scroll(page, scroll_amount)
        scrolled += scroll_amount

        # Dwell on ~30-50% of scroll stops (simulating reading a post)
        if random.random() < random.uniform(0.3, 0.5):
            await asyncio.sleep(random.uniform(1, 5))
        else:
            await asyncio.sleep(human_delay(0.3, 1.0))

    duration = round(time.time() - start, 1)
    return {"type": "feed_scroll", "screen_heights": screen_heights, "duration": duration}
