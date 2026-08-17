#!/usr/bin/env python3
"""Live moat smoke: does evaluate_designer_visually produce a REAL Gemini verdict?

Phase 2, M0.5 (moat-first). Isolates the Designer moat — the VLM portfolio
evaluation that has NEVER run in production — from the full acquisition
pipeline. Fetches one or more real portfolio images, runs the live Gemini
vision eval against a real brief's design rubric, and prints the verdict.
Run before any M1-M6 rewrite: the rewrites assume the moat produces usable
verdicts, and this is the first time it has been exercised live.

    GOOGLE_API_KEY=... .venv/bin/python -m tools.designer_moat_smoke \
        --brief path/to/designer_brief.json \
        --image-url https://example.com/portfolio-cover.jpg [--image-url ...]

PASS = a real, grounded verdict (yes/no/borderline) with per-principle scores
and an EMPTY fallback_reason — i.e. Gemini actually evaluated the images
against the rubric, no hallucination guard fired, sane cost. A fallback_reason
(schema/grounding guard, both-vendors-failed) or a crash means the moat does
not work live yet — fix that first. PASS only means the moat *mechanically*
works; whether the verdict is the RIGHT call for the portfolio is a rubric/
prompt-quality judgment for Sam to eyeball.
"""

from __future__ import annotations

import argparse
import urllib.request

from shared import config
from designer.orchestrator import load_brief_dict, _hydrate_default_rubric
from designer.vision_evaluation import (
    evaluate_designer_visually,
    gemini_vision_llm_call,
)


def _fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ClorisDesigner/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return resp.read(5_000_000)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] fetch failed for {url[:80]}: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Designer moat smoke (Gemini vision eval).")
    ap.add_argument(
        "--brief",
        required=True,
        help="Path to a designer brief JSON. Its design_rubric is used; the default rubric is injected if absent.",
    )
    ap.add_argument(
        "--image-url",
        action="append",
        required=True,
        dest="image_urls",
        help="Portfolio image URL to evaluate. Repeatable for multiple images.",
    )
    ap.add_argument("--name", default="Smoke Candidate", help="Candidate display name for the prompt.")
    ap.add_argument("--headline", default="", help="Candidate headline for the prompt.")
    args = ap.parse_args()

    model = getattr(config, "DESIGNER_VISION_MODEL_NAME", "gemini-3.1-pro-preview")
    if not getattr(config, "GOOGLE_API_KEY", ""):
        print("[smoke] VERDICT: FAIL — GOOGLE_API_KEY is not set; the moat cannot run.")
        return 1

    brief = load_brief_dict(args.brief)
    if "designer" not in (brief.get("target_modules") or []):
        brief.setdefault("target_modules", []).append("designer")
    _hydrate_default_rubric(brief)
    principles = (brief.get("design_rubric") or {}).get("principles") or []
    print(f"[smoke] rubric principles: {len(principles)} | vision model: {model}")

    image_bytes_list: list[bytes] = []
    asset_metadata: list[tuple[str, str, str]] = []
    for url in args.image_urls:
        data = _fetch(url)
        if data is None:
            continue
        image_bytes_list.append(data)
        asset_metadata.append((url, "smoke", ""))
    if not image_bytes_list:
        print("[smoke] VERDICT: FAIL — no portfolio images could be fetched.")
        return 1
    print(f"[smoke] fetched {len(image_bytes_list)} image(s); calling LIVE Gemini vision eval...")

    result = evaluate_designer_visually(
        brief=brief,
        candidate_display_name=args.name,
        candidate_headline=args.headline,
        image_bytes_list=image_bytes_list,
        asset_metadata=asset_metadata,
        vision_llm_call=gemini_vision_llm_call,
        model=model,
    )
    j = result.judgment
    print(f"\n[smoke] verdict        : {j.overall_verdict}")
    print(f"[smoke] confidence     : {j.overall_confidence}")
    print(f"[smoke] fallback_reason: {j.fallback_reason!r}")
    print(f"[smoke] cost_estimate  : ${j.cost_estimate_usd}")
    print(f"[smoke] principles ({len(j.principles)}):")
    for p in j.principles:
        print(f"    - {p.name}: score={p.score} anchor={p.anchor} consistency_pass={p.anchor_consistency_pass}")
        print(f"        {p.reasoning[:160]}")

    clean = (
        j.overall_verdict in {"yes", "no", "borderline"}
        and not j.fallback_reason
        and len(j.principles) > 0
    )
    if clean:
        print(
            f"\n[smoke] VERDICT: PASS — real grounded verdict, no hallucination guard fired. "
            f"Eyeball whether '{j.overall_verdict}' is the right call for this portfolio "
            f"(that is the rubric/prompt-quality question, separate from 'does the moat run')."
        )
        return 0

    print(
        f"\n[smoke] VERDICT: FAIL — fallback_reason={j.fallback_reason!r}, principles={len(j.principles)}; "
        f"the moat did not cleanly evaluate."
    )
    print(
        "[smoke] Triage:\n"
        "  - fallback_reason mentions schema/grounding -> prompt or response parsing is off (vision_evaluation layers 1-2).\n"
        "  - both_vendors_failed -> Gemini API/model issue (check GOOGLE_API_KEY, model name, quota).\n"
        "  - empty principles -> the rubric has no principles (check the brief's design_rubric / config/design-rubrics/default.json)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
