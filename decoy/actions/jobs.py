"""Job browsing — scroll listings, click into a few, scan briefly."""

import asyncio
import json
import random
import time
from shared.human_timing import human_delay, human_delay_correlated
from decoy.actions._utils import ghost_click, human_scroll


async def browse_jobs(page, cursor=None) -> dict:
    """Navigate to /jobs/, scroll listings, click into 1-3 jobs.

    Args:
        page: Playwright page object
        cursor: python_ghost_cursor instance for Bézier click trajectories

    Returns:
        dict with action metadata
    """
    start = time.time()
    jobs_clicked = 0

    # Navigate to jobs via nav bar (ghost-cursor click)
    try:
        jobs_nav = page.locator("#global-nav a[href*='/jobs']").first
        if await jobs_nav.count() > 0:
            await ghost_click(cursor, page, "#global-nav a[href*='/jobs']")
        else:
            await page.goto("https://www.linkedin.com/jobs/", wait_until="domcontentloaded")
    except Exception:
        await page.goto("https://www.linkedin.com/jobs/", wait_until="domcontentloaded")

    await asyncio.sleep(human_delay(1.5, 3.0))

    # Get viewport height
    try:
        vp = json.loads(await page.evaluate(
            "JSON.stringify({w: window.innerWidth, h: window.innerHeight})"
        ))
        vh = vp["h"]
    except Exception:
        vh = 800

    # Scroll through job listings
    scroll_depth = random.randint(2, 5)
    for _ in range(scroll_depth):
        scroll_amount = int(vh * random.uniform(0.4, 0.7))
        await human_scroll(page, scroll_amount)
        await asyncio.sleep(human_delay_correlated(1.5))

    # Click into 1-3 job listings
    target_clicks = random.randint(1, 3)
    for _ in range(target_clicks):
        try:
            cards = await page.locator(
                ".job-card-container, .jobs-search-results__list-item, "
                ".scaffold-layout__list-item"
            ).all()
            visible = []
            for card in cards:
                try:
                    box = await card.bounding_box()
                    if box and 0 < box["y"] < vh:
                        visible.append(card)
                except Exception:
                    continue
            if not visible:
                break

            target = random.choice(visible)
            await ghost_click(cursor, page, target)
            jobs_clicked += 1

            # Dwell on job detail (10-30s with log-normal shape)
            await asyncio.sleep(human_delay(10, 30, mu=2.5, sigma=0.4))

            # Optionally scroll within the job detail
            if random.random() > 0.4:
                await human_scroll(page, random.randint(200, 500))
                await asyncio.sleep(human_delay_correlated(1.5))

            await asyncio.sleep(human_delay(0.5, 2.0))

        except Exception:
            break

    # Return to feed
    await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")

    duration = round(time.time() - start, 1)
    return {"type": "jobs_browse", "jobs_clicked": jobs_clicked, "duration": duration}
