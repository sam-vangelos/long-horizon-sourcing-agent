"""Decoy Agent — passive LinkedIn browsing to generate ambient activity.

Operates on core LinkedIn.com surfaces (feed, notifications, jobs).
Never touches Recruiter. Runs in a separate tab.
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Optional

from decoy.actions.feed import scroll_feed
from decoy.actions.notifications import check_notifications
from decoy.actions.jobs import browse_jobs
from decoy.scheduler import BurstScheduler
from shared.human_timing import human_delay

LOG_DIR = Path.home() / ".sourcing-governor"
DECOY_LOG = LOG_DIR / "decoy.jsonl"

# Action weights: jobs (5) > feed (3) > notifications (1)
ACTIONS = [
    ("feed", scroll_feed, 3),
    ("notifications", check_notifications, 1),
    ("jobs", browse_jobs, 5),
]
TOTAL_WEIGHT = sum(w for _, _, w in ACTIONS)


def _weighted_pick(count: int) -> list:
    """Pick count actions weighted by frequency."""
    picked = []
    for _ in range(count):
        r = random.random() * TOTAL_WEIGHT
        for name, fn, weight in ACTIONS:
            r -= weight
            if r <= 0:
                picked.append((name, fn))
                break
    return picked


def _log_decoy(action: str, meta: dict):
    """Append a decoy log entry."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": action, **meta}
    with open(DECOY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


class DecoyAgent:
    """Manages the decoy tab and executes activity bursts."""

    def __init__(self, browser_context):
        """
        Args:
            browser_context: Playwright BrowserContext (shared with sourcing agent)
        """
        self._context = browser_context
        self._page = None
        self._cursor = None
        self._owns_page = False

    async def _ensure_tab(self):
        """Find or create a linkedin.com tab (non-Recruiter).
        Initializes ghost-cursor for Bézier mouse trajectories.
        """
        if self._page and not self._page.is_closed():
            return

        # Always create a dedicated decoy tab so we never hijack the user's
        # normal LinkedIn browsing or any auxiliary automation tab.
        self._page = await self._context.new_page()
        self._owns_page = True
        await self._page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
        await asyncio.sleep(human_delay(2, 4))
        self._init_cursor()

    def _init_cursor(self):
        """Initialize ghost-cursor for human-like mouse movement on the decoy tab."""
        try:
            from python_ghost_cursor.playwright_async import create_cursor
            self._cursor = create_cursor(self._page)
        except Exception as e:
            self._cursor = None

    async def execute_burst(self) -> list[dict]:
        """Execute one activity burst: 1-3 random actions.

        Returns:
            List of action result dicts
        """
        await self._ensure_tab()

        count = random.randint(1, 3)
        actions = _weighted_pick(count)
        results = []

        action_names = [name for name, _ in actions]
        _log_decoy("burst_start", {"actions": action_names})

        for i, (name, fn) in enumerate(actions):
            try:
                result = await fn(self._page, cursor=self._cursor)
                results.append(result)
            except Exception as e:
                _log_decoy("action_error", {"action": name, "error": str(e)})
                results.append({"type": name, "error": str(e), "duration": 0})

            # Intra-burst delay between actions
            if i < len(actions) - 1:
                await asyncio.sleep(human_delay(0.5, 3.0))

        total_duration = sum(r.get("duration", 0) for r in results)
        summary_parts = [f"{r['type']} ({r.get('duration', 0)}s)" for r in results]
        _log_decoy("burst_end", {"duration": total_duration, "summary": ", ".join(summary_parts)})

        return results

    async def run_dormant_loop(self, stop_event: asyncio.Event, duration_seconds: float):
        """Run the decoy in dormant mode for a specified duration.

        Executes periodic bursts with log-normal inter-burst timing.

        Args:
            stop_event: Set this to stop the loop early
            duration_seconds: Max wall clock to run
        """
        scheduler = BurstScheduler(mode="dormant")
        start = time.time()

        while not stop_event.is_set():
            elapsed = time.time() - start
            if elapsed >= duration_seconds:
                break

            await self.execute_burst()

            if stop_event.is_set():
                break

            interval = scheduler.next_interval()
            remaining = duration_seconds - (time.time() - start)
            wait_time = min(interval, remaining)

            if wait_time <= 0:
                break

            # Wait in 5-second chunks for responsiveness to stop_event
            waited = 0.0
            while waited < wait_time and not stop_event.is_set():
                chunk = min(5.0, wait_time - waited)
                await asyncio.sleep(chunk)
                waited += chunk

    async def close(self):
        """Close the decoy tab if we created it."""
        if self._page and not self._page.is_closed():
            try:
                # Navigate to feed before closing (clean state)
                await self._page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
            except Exception:
                pass
            if self._owns_page:
                try:
                    await self._page.close()
                except Exception:
                    pass
        self._page = None
        self._cursor = None
        self._owns_page = False
