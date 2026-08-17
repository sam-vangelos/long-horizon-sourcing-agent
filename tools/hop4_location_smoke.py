#!/usr/bin/env python3
"""Hop-4 live smoke: verify LinkedInBrowser.apply_location_filter on a real seat.

This is the H4-S6 gate (plans/hop4-structured-filter-execution.md). It applies ONE
Location filter to the CURRENT Recruiter search via the real apply_location_filter
method, confirms the chip landed, then REMOVES it so the search is left unchanged. Run
it on a low-stakes scratch project — it mutates the live search transiently.

    .venv/bin/python -m tools.hop4_location_smoke --location "New York City Metropolitan Area"

Requires a Chrome with remote debugging on (config.CDP_URL), already on a Recruiter
/talent search tab. A PASS proves the captured selectors drive the live DOM, and clears
'locations' to graduate to STABLE_NOW_CONTROLS (the constant + seam-pin flips are
pre-staged). A FAIL prints which selector to adjust in
docs/linkedin-recruiter-dom-map.md (Location row) and linkedin/browser.py.
"""

from __future__ import annotations

import argparse
import asyncio

from linkedin.browser import LinkedInBrowser
from shared.governor import SessionGovernor


async def _run(location: str) -> int:
    # P8.1: real governor — live Recruiter seat, same rule as hop4_title_smoke.
    browser = LinkedInBrowser(governor=SessionGovernor())
    await browser.connect()
    try:
        await browser.require_recruiter_tab()
        before = await browser._peek_results_count_text()
        print(f"[smoke] results before: {before!r}")
        print(f"[smoke] applying location filter: {location!r}")

        ok = await browser.apply_location_filter([location])
        print(f"[smoke] apply_location_filter returned: {ok}")

        rail = browser.page.locator("aside.left-rail")
        # Applied pill: li[data-test-facet-pills-item] containing a dismiss control
        # named "Remove {value}" (2026-06 capture; the legacy [data-test-pill-label]
        # is gone). The dismiss "X" is hover-gated, so confirm the visible pill ITEM
        # (CSS :has matches the button regardless of its visibility state).
        _safe = location.replace("\\", "\\\\").replace('"', '\\"')
        chip = rail.locator(
            f'li[data-test-facet-pills-item]:has('
            f'button[data-test-pill-dismiss][aria-label="Remove {_safe}"])'
        ).first
        chip_present = await chip.is_visible(timeout=2000)
        after = await browser._peek_results_count_text()
        print(f"[smoke] chip present: {chip_present} | results after: {after!r}")

        # Cleanup: remove the chip we added so the scratch search is unchanged. The
        # dismiss "X" only becomes clickable on hover, so hover the pill first.
        removed = False
        if chip_present:
            try:
                await chip.hover()
            except Exception:
                pass
            remove_btn = rail.locator(
                f'button[data-test-pill-dismiss][aria-label="Remove {_safe}"]'
            ).first
            try:
                await browser._ghost_click_locator(remove_btn)
            except Exception:
                pass
            removed = not await chip.is_visible(timeout=2000)
        print(f"[smoke] chip removed (cleanup): {removed}")

        verdict = bool(ok and chip_present and removed)
        print(
            f"\n[smoke] VERDICT: {'PASS' if verdict else 'FAIL'} "
            f"(applied={ok}, chip_seen={chip_present}, cleaned_up={removed})"
        )
        if not verdict:
            print(
                "[smoke] Triage:\n"
                "  - apply=True but no chip  -> option/chip selector is off (the exact-match "
                "option or li[data-test-facet-pills-item] / facet-pill__action[data-test-pill-dismiss])\n"
                "  - apply=False             -> the Add button or editor-input selector is off "
                "(button:has-text('geographic location') / input[placeholder*='location'])\n"
                "  - chip seen but not removed -> the Remove-button selector is off\n"
                "  Adjust in linkedin/browser.py + docs/linkedin-recruiter-dom-map.md (Location)."
            )
        return 0 if verdict else 1
    finally:
        # Release the Playwright/CDP connection without closing Sam's Chrome.
        await browser.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser(description="Hop-4 live smoke for apply_location_filter")
    ap.add_argument(
        "--location",
        default="New York City Metropolitan Area",
        help="Exact LinkedIn metro/location name to apply and then remove",
    )
    args = ap.parse_args()
    return asyncio.run(_run(args.location))


if __name__ == "__main__":
    raise SystemExit(main())
