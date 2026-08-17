#!/usr/bin/env python3
"""Live smoke: verify LinkedInBrowser.apply_title_filter on a real seat.

Sibling of tools/hop4_company_smoke.py and tools/hop4_location_smoke.py. It
applies ONE Job Title filter to the CURRENT Recruiter search via the real
apply_title_filter method, confirms the chip landed, then REMOVES it so the
search is left unchanged. Run on a low-stakes scratch project — it mutates the
live search transiently.

    .venv/bin/python -m tools.hop4_title_smoke --title "Software Engineer"

Optionally set the scope (default: "any" — uses LinkedIn's facet default
"Current or Past"):

    .venv/bin/python -m tools.hop4_title_smoke --title "Software Engineer" --scope any

Requires Chrome with remote debugging on (config.CDP_URL), already on a
Recruiter /talent search tab. A PASS proves the captured selectors AND the
scope driver work on the live DOM, and clears 'job_titles' to graduate to
STABLE_NOW_CONTROLS. A FAIL prints which selector to adjust in
docs/linkedin-recruiter-dom-map.md (Job titles row) and linkedin/browser.py.
"""

from __future__ import annotations

import argparse
import asyncio

from linkedin.browser import LinkedInBrowser
from shared.governor import SessionGovernor


async def _run(title: str, scope: str) -> int:
    # P8.1: this hits a live Recruiter seat, so it gets the real governor
    # like any other production browser construction — even though this
    # smoke never opens a profile, "impossible-by-default" means every
    # non-test construction site is governed, not just the ones that
    # currently happen to call open_profile*.
    browser = LinkedInBrowser(governor=SessionGovernor())
    await browser.connect()
    try:
        await browser.require_recruiter_tab()

        rail = browser.page.locator("aside.left-rail")
        try:
            await rail.first.wait_for(state="attached", timeout=15000)
            on_results_view = True
        except Exception:
            on_results_view = False
        if not on_results_view:
            current = browser.page.url
            print(
                "[smoke] NOT on a results view: the search-results refinement rail "
                "(aside.left-rail) is absent.\n"
                f"  Current page: {current[:120]}\n"
                "  apply_title_filter resolves its editor/chip inside that rail, so it "
                "would fail closed here and MASK a real selector PASS as a miss.\n"
                "  Run a keyword search first so the page is a populated results rail (the "
                "state production always applies from), then re-run this smoke.\n"
                "\n[smoke] VERDICT: FAIL (precondition: not on a results view)"
            )
            return 1

        before = await browser._peek_results_count_text()
        print(f"[smoke] results before: {before!r}")
        print(f"[smoke] applying title filter: {title!r} (scope={scope!r})")

        ok = await browser.apply_title_filter([title], temporal_scope=scope)
        print(f"[smoke] apply_title_filter returned: {ok}")

        # Applied pill: li[data-test-facet-pills-item] containing a dismiss control
        # named "Remove {value}" (2026-06 capture; the legacy [data-test-pill-label]
        # is gone). The dismiss "X" is hover-gated, so confirm the visible pill ITEM
        # (CSS :has matches the button regardless of its visibility state).
        _safe = title.replace("\\", "\\\\").replace('"', '\\"')
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
                "  - apply=False             -> the reveal button, editor-input, or scope "
                "selector is off (button:has-text('Job titles or boolean') / "
                "input[placeholder*='job title'] / [data-test-scope-facet-dropdown-trigger])\n"
                "  - chip seen but not removed -> the Remove-button selector is off\n"
                "  Adjust in linkedin/browser.py + docs/linkedin-recruiter-dom-map.md (Job titles)."
            )
        return 0 if verdict else 1
    finally:
        await browser.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live smoke for apply_title_filter")
    ap.add_argument(
        "--title",
        default="Software Engineer",
        help="Exact LinkedIn job title to apply and then remove",
    )
    ap.add_argument(
        "--scope",
        default="any",
        help="Temporal scope (any = use LinkedIn default 'Current or Past')",
    )
    args = ap.parse_args()
    return asyncio.run(_run(args.title, args.scope))


if __name__ == "__main__":
    raise SystemExit(main())
