#!/usr/bin/env python3
"""Live smoke: verify LinkedInBrowser.apply_company_filter on a real seat.

Sibling of tools/hop4_location_smoke.py. It applies ONE Company filter to the CURRENT
Recruiter search via the real apply_company_filter method, confirms the chip landed,
then REMOVES it so the search is left unchanged. Run it on a low-stakes scratch project
— it mutates the live search transiently.

    .venv/bin/python -m tools.hop4_company_smoke --company "Stripe"

Requires a Chrome with remote debugging on (config.CDP_URL), already on a Recruiter
/talent search tab. A PASS proves the captured selectors drive the live DOM, and clears
'companies' to graduate to STABLE_NOW_CONTROLS (flip the constant + ship the strategist
title/company producer). A FAIL prints which selector to adjust in
docs/linkedin-recruiter-dom-map.md (Companies row) and linkedin/browser.py.
"""

from __future__ import annotations

import argparse
import asyncio

from linkedin.browser import LinkedInBrowser
from shared.governor import SessionGovernor


async def _run(company: str) -> int:
    # P8.1: real governor — live Recruiter seat, same rule as hop4_title_smoke.
    browser = LinkedInBrowser(governor=SessionGovernor())
    await browser.connect()
    try:
        await browser.require_recruiter_tab()

        # R8: results-rail precondition. apply_company_filter resolves its
        # editor + chip inside the search-results refinement rail
        # (aside.left-rail). On Chrome 148 connect() lands a Playwright-created
        # page that can open on /advanced (the entry FORM) or another non-results
        # URL where that rail is absent — there, apply_company_filter fails closed
        # and returns False even though the selectors are correct, MASKING a real
        # PASS as a selector miss. (The earlier location smoke only passed because
        # that seat happened to be on a populated results page.) Production is
        # unaffected: search_mutation reaches apply only after a keyword search has
        # run, so the page is already a populated results rail. Require the rail
        # here and fail the smoke LOUDLY if we are not on a results view, rather
        # than letting a fail-closed False read as a selector problem.
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
                "  apply_company_filter resolves its editor/chip inside that rail, so it "
                "would fail closed here and MASK a real selector PASS as a miss.\n"
                "  Run a keyword search first so the page is a populated results rail (the "
                "state production always applies from), then re-run this smoke.\n"
                "\n[smoke] VERDICT: FAIL (precondition: not on a results view)"
            )
            return 1

        before = await browser._peek_results_count_text()
        print(f"[smoke] results before: {before!r}")
        print(f"[smoke] applying company filter: {company!r}")

        ok = await browser.apply_company_filter([company])
        print(f"[smoke] apply_company_filter returned: {ok}")

        rail = browser.page.locator("aside.left-rail")
        # Applied pill: li[data-test-facet-pills-item] containing a dismiss control
        # named "Remove {value}" (2026-06 capture; the legacy [data-test-pill-label]
        # is gone). The dismiss "X" is hover-gated, so confirm the visible pill ITEM
        # (CSS :has matches the button regardless of its visibility state).
        _safe = company.replace("\\", "\\\\").replace('"', '\\"')
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
                "(button:has-text('Companies or boolean') / input[placeholder*='company'])\n"
                "  - chip seen but not removed -> the Remove-button selector is off\n"
                "  Adjust in linkedin/browser.py + docs/linkedin-recruiter-dom-map.md (Companies)."
            )
        return 0 if verdict else 1
    finally:
        # Release the Playwright/CDP connection without closing Sam's Chrome.
        await browser.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live smoke for apply_company_filter")
    ap.add_argument(
        "--company",
        default="Stripe",
        help="Exact LinkedIn company name to apply and then remove",
    )
    args = ap.parse_args()
    return asyncio.run(_run(args.company))


if __name__ == "__main__":
    raise SystemExit(main())
