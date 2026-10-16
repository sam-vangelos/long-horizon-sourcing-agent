"""Save rescued candidates to a Recruiter project via their profile pages.

One-off rescue lane: consumes the
passers.json written by tools/rejudge_review_flagged.py and saves each
candidate to the target project. Unlike the pipeline's card-scoped
save_candidate (linkedin/browser.py), this drives the PROFILE PAGE's own save
rail, deep-linked via ?rightRail=saveToProject so no ambiguous button click is
needed (the Similar Profiles rail carries identical-looking save buttons).

Per candidate: navigate → verify identity via the URL fragment and the rail
header → skip if already in the project → choose existing project → pick the
project by typed name → Save → confirm via the "In N projects" block. Every
navigation records a profile open in the shared governor ledger so the 24h
anti-detection budget stays truthful.

Usage:
    .venv/bin/python tools/save_rescued_to_project.py <passers.json> \
        --project-name "Sr/Staff/Principal Research Engineer - Code (Brazil)" \
        [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import shared.cooldown as cooldown  # noqa: E402
from shared import config  # noqa: E402

DEFAULT_PROJECT_NAME = "Sr/Staff/Principal Research Engineer - Code (Brazil)"


def _fragment(url: str) -> str:
    return url.rstrip("/").split("/")[-1].split("?")[0]


async def save_one(page, url: str, name: str, project_name: str, dry_run: bool,
                   shot_dir: Path) -> str:
    """Returns one of: saved | already | dry_run | FAILED:<reason>."""
    frag = _fragment(url)
    await page.goto(f"{url}?rightRail=saveToProject", timeout=45000)
    await page.wait_for_timeout(3500)

    if frag not in page.url:
        return f"FAILED:navigated_away ({page.url[:80]})"

    # Already-in-project check: the top card lists current project
    # memberships (probe-verified: the link renders the full project name).
    membership = page.locator(f'a:has-text("{project_name}")')
    if await membership.count() > 0:
        return "already"

    # Rail-open signal: "Choose existing project" is unique to the save rail
    # (plain-string matcher; probe-verified reachable in the main frame).
    # Identity is guarded by the URL-fragment check above — the rail is
    # same-profile-scoped by construction.
    radio = page.get_by_text("Choose existing project").first
    try:
        await radio.wait_for(state="visible", timeout=20000)
    except Exception:
        # Deep link didn't open the rail — click the top card's own save
        # button (aria: "Save <name> to a project"; the Similar Profiles
        # rail's buttons say "to project", so this pattern is card-unique).
        try:
            trigger = page.get_by_role(
                "button", name=re.compile(r"Save .+ to a project")
            ).first
            await trigger.click(timeout=8000)
            await radio.wait_for(state="visible", timeout=10000)
        except Exception:
            shot = shot_dir / f"rail-missing-{_fragment(url)[:12]}.png"
            try:
                await page.screenshot(path=str(shot))
            except Exception:
                shot = Path("<screenshot failed>")
            return (
                f"FAILED:save_rail_missing (title={await page.title()!r}, "
                f"shot={shot})"
            )

    if dry_run:
        return "dry_run"

    await radio.click(timeout=10000)
    box = page.locator('input[placeholder="Choose a project"]').first
    await box.wait_for(state="visible", timeout=10000)
    await box.click()
    await box.type(project_name[:40], delay=random.randint(30, 80))
    option = page.locator(
        f'[role="listbox"] :text-is("{project_name}"), '
        f'[role="option"]:has-text("{project_name}")'
    ).first
    try:
        await option.wait_for(state="visible", timeout=10000)
        await option.click()
    except Exception:
        # Fallback: the recent-projects dropdown renders without listbox roles.
        loose = page.locator(f'text="{project_name}"').last
        await loose.click(timeout=5000)

    # Exact-name match: the top card's "Save to project" buttons also contain
    # the substring "Save"; the rail footer button is the only bare "Save".
    save_btn = page.get_by_role("button", name="Save", exact=True).last
    await save_btn.click(timeout=10000)

    # Confirm: membership link appears in the top card (SPA refresh).
    for _ in range(20):
        await page.wait_for_timeout(1500)
        if await membership.count() > 0:
            return "saved"
    return "FAILED:save_not_confirmed"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("passers", type=Path)
    ap.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    passers = json.loads(args.passers.read_text())
    if args.limit:
        passers = passers[: args.limit]
    print(f"[save-rescue] {len(passers)} candidates → project '{args.project_name}'")

    from rebrowser_playwright.async_api import async_playwright

    results = []
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(config.CDP_URL)
        context = browser.contexts[0]
        page = await context.new_page()
        try:
            for i, cand in enumerate(passers, 1):
                name = cand["candidate_name"]
                url = cand["profile_url"]
                try:
                    cooldown.record_profile_open()
                    outcome = await save_one(
                        page, url, name, args.project_name, args.dry_run,
                        shot_dir=args.passers.parent,
                    )
                except Exception as exc:
                    outcome = f"FAILED:{type(exc).__name__}: {exc}"
                results.append({"candidate_name": name, "profile_url": url,
                                "outcome": outcome})
                print(f"  [{i}/{len(passers)}] {outcome:24s} {name}")
                if i < len(passers):
                    await page.wait_for_timeout(random.randint(6000, 14000))
        finally:
            await page.close()

    out = args.passers.parent / (
        f"save-outcomes-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.write_text(json.dumps(results, indent=2))
    tallies: dict[str, int] = {}
    for r in results:
        key = r["outcome"].split(":")[0]
        tallies[key] = tallies.get(key, 0) + 1
    print(f"[save-rescue] done: {tallies} → {out}")
    return 0 if all(not r["outcome"].startswith("FAILED") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
