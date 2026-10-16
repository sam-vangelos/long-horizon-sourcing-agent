#!/usr/bin/env python3
"""Preview or live-smoke one saved candidate's production note primitive.

    .venv/bin/python tools/smoke_candidate_note.py \
        --profile-url <recruiter profile url> \
        --project-id 9000000010 \
        --state-dir output/state/linkedin/9000000010-prre-brazil-20260812 \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from linkedin.candidate_note import build_candidate_note, note_probe_text  # noqa: E402
from linkedin.input_backends import build_note_typing_plan  # noqa: E402
from shared.schemas import OpusDecision, SearchString  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required projection not found: {path}")
    rows: list[dict] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected an object in {path}:{line_number}")
        rows.append(row)
    return rows


def _load_note_context(
    state_dir: Path,
    profile_url: str,
) -> tuple[OpusDecision, SearchString]:
    requested_url = profile_url.strip()
    verdicts = [
        row
        for row in _read_jsonl(state_dir / "final_judgments.jsonl")
        if str(row.get("profile_url") or "").strip() == requested_url
    ]
    if not verdicts:
        raise ValueError(
            f"no final_judgments.jsonl row exactly matches profile_url={requested_url!r}"
        )

    snippets = [
        row
        for row in _read_jsonl(state_dir / "snippets.jsonl")
        if str(row.get("profile_url") or "").strip() == requested_url
    ]
    if not snippets:
        raise ValueError(
            f"no snippets.jsonl row exactly matches profile_url={requested_url!r}"
        )
    snippet = snippets[-1]
    try:
        string_id = int(snippet["source_string_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("matching snippet has no valid source_string_id") from exc

    return OpusDecision.from_dict(verdicts[-1]), SearchString(
        id=string_id,
        name=str(snippet.get("source_string_name") or f"string-{string_id}"),
        boolean="",
    )


def _estimated_typing_seconds(note_text: str) -> float:
    plan = build_note_typing_plan(note_text, rng=random.Random(0))
    return sum(step.delay_seconds for step in plan.steps)


def _print_preview(note_text: str) -> None:
    print(f"note_chars: {len(note_text)}")
    print(f"estimated_typing_seconds: {_estimated_typing_seconds(note_text):.1f}")
    print("intended_note:")
    print(note_text)


async def _scan_for_fragment(page, requested_fragment: str):
    """Return (match_or_None, fragments_seen). Never clicks."""
    from linkedin.browser import LinkedInBrowser

    links = page.locator('a[href*="/profile/"]')
    seen: list[str] = []
    for index in range(await links.count()):
        link = links.nth(index)
        href = str(await link.get_attribute("href") or "")
        fragment = LinkedInBrowser._profile_url_fragment(href)
        if not fragment:
            continue
        seen.append(fragment)
        if fragment != requested_fragment:
            continue
        if await link.is_visible(timeout=1000):
            return link, seen
    return None, seen


async def _find_exact_profile_anchor(page, requested_fragment: str):
    """Poll and scroll a lazily-rendered pipeline list for one exact candidate.

    The pipeline is a virtualised Ember list: a fixed sleep after
    domcontentloaded races the rows into existence, and rows below the fold are
    not in the DOM at all until scrolled. Poll first, then scroll in steps,
    re-scanning after each — and report what WAS seen on failure so a
    wrong-page/wrong-sort problem is not reported as a missing candidate.
    """
    seen: list[str] = []
    for _ in range(20):  # ~20s of settling
        match, seen = await _scan_for_fragment(page, requested_fragment)
        if match is not None:
            return match
        if seen:
            break  # list has rendered; stop waiting, start scrolling
        await page.wait_for_timeout(1000)

    for _ in range(12):  # walk the page, re-scanning as rows mount
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(700)
        match, scrolled_seen = await _scan_for_fragment(page, requested_fragment)
        if match is not None:
            return match
        seen = scrolled_seen or seen

    raise RuntimeError(
        "candidate anchor not found by exact identity fragment "
        f"{requested_fragment!r}; {len(seen)} profile anchor(s) were visible "
        f"on this page (first few: {seen[:5]}). If the count is non-zero the "
        "candidate is on another pipeline page or the sort moved them."
    )


async def _check_preconditions(browser, requested_fragment: str) -> str:
    from linkedin.browser import PROFILE_PANEL_SELECTOR

    actual_fragment = browser.current_profile_identity_fragment()
    if actual_fragment != requested_fragment:
        raise RuntimeError(
            "profile identity precondition failed: "
            f"requested={requested_fragment!r}, current={actual_fragment!r}"
        )

    panels = browser.page.locator(PROFILE_PANEL_SELECTOR)
    panel_count = await panels.count()
    if panel_count != 1:
        raise RuntimeError(
            f"profile panel precondition failed: expected 1, found {panel_count}"
        )
    panel = panels.first
    await panel.wait_for(state="visible", timeout=10000)

    # The note route is the candidate's own top-card overflow menu, not the
    # right-rail note-list button (which sits inside a collapsed accordion).
    from linkedin.browser import _PROFILE_PANEL_SCOPE

    selector = "button[data-test-more-actions-trigger]"
    global_button_count = await browser.page.locator(selector).count()
    panel_button_count = await browser.page.locator(
        _PROFILE_PANEL_SCOPE + " " + selector
    ).count()
    # Only the PANEL-scoped count is a precondition: every pipeline row carries
    # its own more-actions trigger, so a page-wide count of 1 is never true here.
    if panel_button_count != 1:
        raise RuntimeError(
            "more-actions trigger precondition failed: expected exactly one "
            "trigger inside the profile panel, found "
            f"inside_panel={panel_button_count} (page-wide={global_button_count})"
        )

    titles = panel.locator("[data-test-note-list-title]")
    title_count = await titles.count()
    if title_count != 1:
        raise RuntimeError(
            f"notes title precondition failed: expected 1, found {title_count}"
        )
    title = str(await titles.first.inner_text(timeout=2000)).strip()
    if re.fullmatch(r"Notes\s*\(\d+\)", title) is None:
        raise RuntimeError(f"notes title precondition failed: unreadable {title!r}")
    return title


async def _read_notes_title(browser) -> str:
    from linkedin.browser import PROFILE_PANEL_SELECTOR

    panels = browser.page.locator(PROFILE_PANEL_SELECTOR)
    if await panels.count() != 1:
        raise RuntimeError("profile panel does not resolve exactly once")
    titles = panels.first.locator("[data-test-note-list-title]")
    if await titles.count() != 1:
        raise RuntimeError("notes title does not resolve exactly once")
    return str(await titles.first.inner_text(timeout=2000)).strip()


async def _read_landed_note(browser, note_text: str) -> str:
    from linkedin.browser import PROFILE_PANEL_SELECTOR

    panels = browser.page.locator(PROFILE_PANEL_SELECTOR)
    if await panels.count() != 1:
        raise RuntimeError("profile panel does not resolve exactly once")
    panel = panels.first

    # Use the SAME notes-scoped selectors and visible-only rule as the
    # primitive: a profile-body expander also matches the generic attribute,
    # and the notes expander stays mounted-but-hidden when nothing is collapsed.
    from linkedin.browser import LinkedInBrowser as _LB

    expand = browser.page.locator(_LB._NOTE_EXPAND_PRIMARY)
    expand_count = await expand.count()
    if expand_count == 0:
        expand = browser.page.locator(_LB._NOTE_EXPAND_FALLBACK)
        expand_count = await expand.count()
    if expand_count > 1:
        raise RuntimeError(f"notes expand control is ambiguous ({expand_count})")
    if expand_count == 1 and await expand.first.is_visible():
        await browser._ghost_click_locator(expand.first)
        await browser.page.wait_for_timeout(250)

    # The note carries no marker, so the landed text is found by the note's own
    # opening line — the same probe the primitive writes with.
    probe = note_probe_text(note_text)
    if not probe:
        raise RuntimeError("cannot read a landed note for an empty intended note")
    texts = await panel.locator("*").evaluate_all(
        """(elements, probe) => elements
            .map((element) => (element.innerText || '').trim())
            .filter((text) => text.includes(probe))""",
        probe,
    )
    candidates = []
    for value in texts:
        text = str(value or "").strip()
        if probe not in text:
            continue
        candidates.append(text[text.index(probe) :].strip())
    if not candidates:
        raise RuntimeError("no landed note text is readable")
    # Prefer the smallest element that carries the WHOLE intended note; the
    # rail also renders the opening line as its own element, and a truncated
    # landing must still be reported so the caller can flag the mismatch.
    complete = [text for text in candidates if note_text in text]
    return min(complete or candidates, key=len)


async def _diagnostic_read(reader, browser) -> str:
    try:
        return await reader(browser)
    except Exception as exc:
        return f"<unreadable: {type(exc).__name__}: {exc}>"


async def _run_live(
    *,
    profile_url: str,
    project_id: str,
    decision: OpusDecision,
    note_text: str,
) -> int:
    from linkedin.browser import PROFILE_PANEL_SELECTOR, LinkedInBrowser
    from shared.governor import SessionGovernor

    requested_fragment = LinkedInBrowser._profile_url_fragment(profile_url)
    if not requested_fragment:
        raise ValueError("--profile-url has no supported Recruiter identity fragment")

    governor = SessionGovernor()
    can_start, reason = governor.can_start_session()
    if not can_start:
        raise RuntimeError(f"governor refused the smoke: {reason}")
    governor.start_session()

    browser = LinkedInBrowser(governor=governor)
    browser.set_required_project_id(project_id)
    connected = False
    result = False
    error: BaseException | None = None
    error_traceback = None
    try:
        await browser.connect()
        connected = True
        try:
            pipeline_url = (
                f"https://www.linkedin.com/talent/hire/{project_id}/manage/all"
            )
            await browser.page.goto(
                pipeline_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )
            await browser.page.wait_for_timeout(1500)

            anchor = await _find_exact_profile_anchor(
                browser.page,
                requested_fragment,
            )
            await browser._ghost_click_locator(
                anchor,
                before_click=governor.check_profile_open_or_raise,
            )
            governor.record_profile_open()
            await browser.page.locator(PROFILE_PANEL_SELECTOR).first.wait_for(
                state="visible",
                timeout=10000,
            )
            # The drawer's action row renders after the panel becomes visible.
            # Scope the wait INSIDE the panel: every pipeline row carries its
            # own more-actions trigger, so an unscoped wait is satisfied by a
            # row while the drawer has not rendered its own yet.
            from linkedin.browser import _PROFILE_PANEL_SCOPE

            await browser.page.locator(
                _PROFILE_PANEL_SCOPE + " button[data-test-more-actions-trigger]"
            ).first.wait_for(state="visible", timeout=30000)

            before_title = await _check_preconditions(browser, requested_fragment)
            _print_preview(note_text)
            print(f"pre_notes: {before_title}")
            result = await browser.write_candidate_note(
                note_text,
                expected_name=decision.candidate_name,
            )
        except BaseException as exc:
            error = exc
            error_traceback = exc.__traceback__
        finally:
            post_title = await _diagnostic_read(_read_notes_title, browser)
            landed_text = await _diagnostic_read(
                lambda target: _read_landed_note(target, note_text),
                browser,
            )
            skipped = (
                result
                and browser._last_note_failure_reason == "note_already_present"
            )
            # A duplicate-guard skip is the guard doing its job — there is no
            # newly landed note to diff, so it is success, not a mismatch.
            mismatch = (not skipped) and landed_text != note_text
            print(f"result: {result}")
            print(f"failure_reason: {browser._last_note_failure_reason!r}")
            print(f"post_notes: {post_title}")
            print(
                "landed_note: "
                + json.dumps(landed_text, ensure_ascii=False)
            )
            if skipped:
                print(
                    "SKIPPED: a note already opening with this evaluation was "
                    "found; the duplicate guard correctly declined to write a "
                    "second one"
                )
            elif mismatch:
                print("MISMATCH: LANDED NOTE DOES NOT MATCH INTENDED NOTE")
            else:
                print("MATCH: landed note is byte-identical to the intended note")

        if error is not None:
            raise error.with_traceback(error_traceback)
        return 0 if result and not mismatch else 1
    finally:
        if connected:
            await browser.disconnect()
        governor.end_session()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview or live-smoke one saved candidate note"
    )
    parser.add_argument("--profile-url", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render the stored note without connecting to a browser",
    )
    args = parser.parse_args(argv)

    decision, search_string = _load_note_context(
        args.state_dir,
        args.profile_url,
    )
    note_text = build_candidate_note(
        decision,
        search_string,
        now=datetime.now(timezone.utc),
    )
    if args.dry_run:
        _print_preview(note_text)
        return 0
    return asyncio.run(
        _run_live(
            profile_url=args.profile_url,
            project_id=args.project_id,
            decision=decision,
            note_text=note_text,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
