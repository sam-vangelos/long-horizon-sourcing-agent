"""Best-effort page capture for LinkedIn anti-detection diagnostics."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_CAPTURE_DIRS = 200
_logged_cap_refusals: set[str] = set()


def _sanitize_reason(reason: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", reason)


def _count_capture_dirs(captures_dir: Path) -> int:
    if not captures_dir.is_dir():
        return 0
    return sum(1 for entry in captures_dir.iterdir() if entry.is_dir())


async def capture_page_state(
    browser: Any,
    state_dir: Path,
    *,
    reason: str,
    run_id: int | None = None,
) -> dict[str, Any] | None:
    try:
        captures_dir = state_dir / "captures"

        if _count_capture_dirs(captures_dir) > _MAX_CAPTURE_DIRS:
            key = str(state_dir)
            if key not in _logged_cap_refusals:
                _logged_cap_refusals.add(key)
                logger.warning(
                    "Refusing page capture: %s already has more than %d capture directories",
                    captures_dir,
                    _MAX_CAPTURE_DIRS,
                )
            return None

        page = browser.page

        url = ""
        try:
            url = page.url
        except Exception:
            pass

        title = ""
        try:
            title = await page.title()
        except Exception:
            pass

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        safe_reason = _sanitize_reason(reason)
        capture_dir = captures_dir / f"{ts}-{safe_reason}"
        capture_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        meta: dict[str, Any] = {
            "url": url,
            "title": title,
            "reason": reason,
            "run_id": run_id,
            "timestamp": timestamp,
        }

        (capture_dir / "meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        try:
            body_text = await page.locator("body").inner_text(timeout=500)
            (capture_dir / "body.txt").write_text(
                body_text[:20000], encoding="utf-8"
            )
        except Exception:
            pass

        try:
            await page.screenshot(
                path=capture_dir / "screenshot.png", full_page=False, timeout=2000
            )
        except Exception:
            pass

        return meta
    except Exception:
        return None
