#!/usr/bin/env python3
"""Run structural rule checkers against captured surface facts.

Loads captures from output/audits/<timestamp>/captures.json (or --audit-dir),
runs every rule checker, writes results to <audit_dir>/rule-results.json.

Rule checkers are pure functions: (capture, api_status) -> list[RuleResult].
This separation keeps the walker (I/O-heavy) decoupled from the validators
(easy to unit-test on synthetic facts).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_common import (
    AUDIT_ROOT,
    ElementFact,
    RuleResult,
    SurfaceCapture,
    audit_dir,
    latest_audit_dir,
    to_dict,
)


# Raw API enum strings that should never appear in editorial body text.
# Source: shared/runtime_state/store.py status / stop_reason enums.
RAW_ENUM_STRINGS = [
    "governor_limit_reached",
    "browser_disconnect_unrecovered",
    "fatal_runtime_error",
    "worker_missing",
    "stop_reason",
    "state_dir",
    "state_key",
    "brief_id",
    "run_id",
    "worker_state",
    "attempt_health",
    "work_unit_progress",
]

# Voice-register metaphors. Used by R21 to ensure they don't bleed into
# operational copy zones (nav, button labels, status pills, page titles).
# Phase 2B fills this in further; for now we seed the obvious ones.
VOICE_METAPHORS = [
    "front of file",
    "front of the file",
    "filed away",
    "filed-away",
    "look through",
    "card-file",
    "look back",
    "pick up",
]

# Operational copy zones — element classes / roles where voice metaphors
# violate R21. We treat any element with these classes as operational.
OPERATIONAL_ZONE_CLASSES = {
    # nav / link / button surfaces
    "homescreen-look-through",
    "card-action-stop",
    "card-action-resume",
    "card-action-archive",
    "splash-mark",
    # eyebrow / section labels
    "section-num",
    "homescreen-section-eyebrow",
    "card-status",
}


def _facts_with_text_transform(facts: list[ElementFact], transform: str) -> list[ElementFact]:
    return [f for f in facts if (f.text_transform or "none").lower() == transform.lower()]


def _is_pure_id_text(text: str) -> bool:
    """True if text reads as a raw identifier (numeric ID / hash / path / state_key)."""
    t = text.strip()
    if not t:
        return False
    # Pure numeric (e.g., "3000000007")
    if re.fullmatch(r"\d{4,}", t):
        return True
    # Numeric with simple suffix (e.g., "3000000006 Clean") — flag as ID-shaped
    if re.match(r"^\d{6,}\b", t):
        return True
    # Hash-like (8+ hex chars, no spaces)
    if re.fullmatch(r"[0-9a-f]{8,}", t):
        return True
    # Filesystem path
    if "/Users/" in t or t.startswith("/") or "output/state/" in t:
        return True
    return False


# ---- Rule checkers ----------------------------------------------------------

def check_R2_no_raw_ids_in_titles(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """R2: page titles, card titles, primary list-row labels are never raw IDs."""
    out: list[RuleResult] = []
    for f in cap.facts:
        if f.tag not in ("h1", "h2"):
            continue
        if not f.is_visible:
            continue
        if _is_pure_id_text(f.text):
            out.append(
                RuleResult(
                    rule_id="R2",
                    class_id="2",
                    surface_slug=cap.slug,
                    viewport_w=cap.viewport_w,
                    severity="high",
                    passed=False,
                    evidence=f"<{f.tag} {f.selector}> = {f.text!r}",
                )
            )
    return out


def check_R9_no_filesystem_paths(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """R9: no /Users/ or output/state/ paths anywhere in editorial body."""
    out: list[RuleResult] = []
    for f in cap.facts:
        if not f.is_visible:
            continue
        if "/Users/" in f.text or "output/state/" in f.text:
            out.append(
                RuleResult(
                    rule_id="R9",
                    class_id="5",
                    surface_slug=cap.slug,
                    viewport_w=cap.viewport_w,
                    severity="medium",
                    passed=False,
                    evidence=f"<{f.tag} {f.selector}> = {f.text[:120]!r}",
                )
            )
    return out


def check_R17_mono_caps_floor(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """R17: any text-transform: uppercase element must be ≥ 14px (0.875rem)."""
    out: list[RuleResult] = []
    for f in _facts_with_text_transform(cap.facts, "uppercase"):
        if not f.is_visible or not f.text:
            continue
        if f.font_size_px > 0 and f.font_size_px < 14.0:
            out.append(
                RuleResult(
                    rule_id="R17",
                    class_id="4",
                    surface_slug=cap.slug,
                    viewport_w=cap.viewport_w,
                    severity="medium",
                    passed=False,
                    evidence=f"<{f.tag} {f.selector}> = {f.text[:60]!r} at {f.font_size_px}px",
                )
            )
    return out


def check_R24_canonical_wording(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """R24: raw API enum strings never appear in editorial body."""
    out: list[RuleResult] = []
    for f in cap.facts:
        if not f.is_visible:
            continue
        # Reference Slip is allowed to surface raw values.
        if any(c.startswith("reference-slip") for c in f.classes):
            continue
        for enum in RAW_ENUM_STRINGS:
            if enum in f.text:
                out.append(
                    RuleResult(
                        rule_id="R24",
                        class_id="6",
                        surface_slug=cap.slug,
                        viewport_w=cap.viewport_w,
                        severity="medium",
                        passed=False,
                        evidence=f"<{f.tag} {f.selector}> contains raw enum {enum!r}",
                    )
                )
                break
    return out


def check_R21_register_separation(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """R21: voice metaphors don't appear in operational copy zones."""
    out: list[RuleResult] = []
    for f in cap.facts:
        if not f.is_visible:
            continue
        # Detect operational zone by class membership OR by tag (button/a/h1).
        is_operational = (
            f.tag in ("button",) or
            any(c in OPERATIONAL_ZONE_CLASSES for c in f.classes)
        )
        if not is_operational:
            continue
        text_lower = f.text.lower()
        for metaphor in VOICE_METAPHORS:
            if metaphor in text_lower:
                out.append(
                    RuleResult(
                        rule_id="R21",
                        class_id="6",
                        surface_slug=cap.slug,
                        viewport_w=cap.viewport_w,
                        severity="medium",
                        passed=False,
                        evidence=f"operational <{f.tag} {f.selector}> = {f.text!r} contains voice metaphor {metaphor!r}",
                    )
                )
                break
    return out


def check_class_5_api_state_dir_leak(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """Class 5 (R9 at the source): /api/status leaks absolute filesystem paths.

    Run once per audit (gated on home@*). Counts every entry whose state_dir
    starts with /Users or otherwise looks absolute.
    """
    out: list[RuleResult] = []
    if not cap.slug.startswith("home"):
        return out
    entries = api_status.get("entries", [])
    leaked = [e for e in entries
              if isinstance(e.get("state_dir"), str)
              and (e["state_dir"].startswith("/Users") or e["state_dir"].startswith("/"))]
    if leaked:
        out.append(
            RuleResult(
                rule_id="R9-API",
                class_id="5",
                surface_slug="api/status",
                viewport_w=0,
                severity="medium",
                passed=False,
                evidence=(
                    f"{len(leaked)}/{len(entries)} entries leak absolute "
                    f"state_dir paths (e.g. {leaked[0]['state_dir'][:90]!r})"
                ),
            )
        )
    return out


def check_class_2_title_collisions(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """Class 2: detect when distinct briefs render with the same primary title.

    Only fires once per audit (gated on home). Walks /api/status entries,
    derives what the title resolver would produce, and flags duplicates.

    Mirrors `cloris/frontend/src/lib/state.ts:resolveRecruiterTitlesWithCollisions`:
    when N>1 entries share the same `brief_role_title` primary, the
    frontend appends `#<state_key>` as a mono-caps subtitle so each
    card disambiguates visually. The audit applies the same two-pass
    logic — collisions are only flagged when the disambiguated forms
    STILL collide (which they shouldn't, since state_keys are unique).
    """
    out: list[RuleResult] = []
    if not cap.slug.startswith("home"):
        return out
    entries = api_status.get("entries", [])

    def derive_title_primary(e: dict[str, Any]) -> str:
        """Approximate the single-entry resolver's primary line."""
        if e.get("brief_role_title"):
            return e["brief_role_title"]
        sk = (e.get("state_key", "") or "").strip()
        # Slug-shaped: starts with a letter and has at least one letter.
        if sk and sk[0].isalpha() and any(c.isalpha() for c in sk):
            return sk.replace("_", " ").replace("-", " ").title()
        # Source-typed generic + disambiguator subtitle. The subtitle is
        # what makes two pure-numeric state_keys render distinctly even
        # though the primary line matches.
        source_label = "LinkedIn search" if e.get("source") == "linkedin" else "GitHub search"
        return f"{source_label} #{sk}" if sk else source_label

    def has_disambiguating_fallback(e: dict[str, Any]) -> bool:
        """Did `derive_title_primary` already encode a state_key suffix?

        The source-typed generic branch returns ``"<source> search #<sk>"``
        which already disambiguates without our help; only the
        ``brief_role_title`` branch is collision-prone.
        """
        if e.get("brief_role_title"):
            return False
        sk = (e.get("state_key", "") or "").strip()
        if sk and sk[0].isalpha() and any(c.isalpha() for c in sk):
            return False
        return True

    primary_counts: dict[str, int] = {}
    for e in entries:
        if not e.get("latest_run"):
            continue
        primary_counts[derive_title_primary(e)] = (
            primary_counts.get(derive_title_primary(e), 0) + 1
        )

    def render_title(e: dict[str, Any]) -> str:
        """The string the recruiter actually sees, post-disambiguation."""
        primary = derive_title_primary(e)
        if primary_counts.get(primary, 0) < 2:
            return primary
        if has_disambiguating_fallback(e):
            return primary
        sk = (e.get("state_key", "") or "").strip()
        if not sk:
            return primary
        return f"{primary} #{sk}"

    title_to_keys: dict[str, list[str]] = {}
    for e in entries:
        if not e.get("latest_run"):
            continue
        t = render_title(e)
        title_to_keys.setdefault(t, []).append(f"{e['source']}/{e['state_key']}")
    for title, keys in title_to_keys.items():
        if len(keys) < 2:
            continue
        out.append(
            RuleResult(
                rule_id="R2-COLLISION",
                class_id="2",
                surface_slug="api/status",
                viewport_w=0,
                severity="high",
                passed=False,
                evidence=(
                    f"{len(keys)} distinct briefs render as title {title!r}: "
                    f"{', '.join(keys[:5])}"
                ),
            )
        )
    return out


def check_class_1_zombie_runs(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """Class 1: API-level — flag any latest_run.status='running' with worker_state='missing'."""
    out: list[RuleResult] = []
    if cap.slug.split("@")[0] != "home":
        # Only check once per audit, not per surface.
        return out
    entries = api_status.get("entries", [])
    for e in entries:
        lr = e.get("latest_run") or {}
        if lr.get("status") == "running" and e.get("worker_state") in ("missing", "stale"):
            started = lr.get("started_at", "?")
            out.append(
                RuleResult(
                    rule_id="CLASS-1-ZOMBIE",
                    class_id="1",
                    surface_slug="api/status",
                    viewport_w=0,
                    severity="critical",
                    passed=False,
                    evidence=(
                        f"{e['source']}/{e['state_key']} run #{lr.get('id')} "
                        f"status=running worker_state={e['worker_state']} started={started}"
                    ),
                )
            )
    return out


def check_class_0_placeholder(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """Class 0: surface contains a literal 'placeholder' / 'not implemented' string."""
    out: list[RuleResult] = []
    placeholders = ["placeholder", "not implemented", "coming soon", "TBD", "TODO"]
    for f in cap.facts:
        if not f.is_visible:
            continue
        text_lower = f.text.lower()
        for phrase in placeholders:
            if phrase.lower() in text_lower:
                out.append(
                    RuleResult(
                        rule_id="CLASS-0-PLACEHOLDER",
                        class_id="0",
                        surface_slug=cap.slug,
                        viewport_w=cap.viewport_w,
                        severity="high",
                        passed=False,
                        evidence=f"<{f.tag} {f.selector}> = {f.text[:120]!r}",
                    )
                )
                break
    return out


def check_class_3_excessive_height(
    cap: SurfaceCapture, api_status: dict[str, Any]
) -> list[RuleResult]:
    """Class 3: page is too tall — inferred from full-page screenshot file size as proxy.

    A page > 8000px tall almost certainly indicates an IA failure (no grouping /
    collapse / pagination). We can't read screenshot dimensions without PIL,
    but we approximate via file size: PNGs over ~2.5MB at scale-2 typically
    indicate excessive vertical content.
    """
    out: list[RuleResult] = []
    if not cap.full_page_screenshot:
        return out
    full_path = (Path(cap.full_page_screenshot)
                 if Path(cap.full_page_screenshot).is_absolute()
                 else AUDIT_ROOT / "_resolve_relative")  # lazy resolution
    # Resolve relative to capture's audit dir (passed via context — we keep
    # it simple here and skip the heuristic if path isn't accessible).
    return out  # (Heuristic disabled until we plumb the audit-dir context.)


# Registry — top-level dispatch for the report.
RULE_CHECKERS: list[Callable[[SurfaceCapture, dict[str, Any]], list[RuleResult]]] = [
    check_R2_no_raw_ids_in_titles,
    check_R9_no_filesystem_paths,
    check_R17_mono_caps_floor,
    check_R24_canonical_wording,
    check_R21_register_separation,
    check_class_1_zombie_runs,
    check_class_2_title_collisions,
    check_class_5_api_state_dir_leak,
    check_class_0_placeholder,
    check_class_3_excessive_height,
]


def load_captures(audit_path: Path) -> tuple[list[SurfaceCapture], dict[str, Any]]:
    raw_caps = json.loads((audit_path / "captures.json").read_text())
    caps: list[SurfaceCapture] = []
    for r in raw_caps:
        facts = [ElementFact(**f) for f in r.pop("facts", [])]
        cap = SurfaceCapture(**r, facts=facts)
        caps.append(cap)
    api_status = json.loads((audit_path / "api-status.json").read_text())
    return caps, api_status


def run_all_checks(captures: list[SurfaceCapture], api_status: dict[str, Any]) -> list[RuleResult]:
    results: list[RuleResult] = []
    for cap in captures:
        for checker in RULE_CHECKERS:
            try:
                results.extend(checker(cap, api_status))
            except Exception as e:
                results.append(
                    RuleResult(
                        rule_id=checker.__name__,
                        class_id="M1",
                        surface_slug=cap.slug,
                        viewport_w=cap.viewport_w,
                        severity="low",
                        passed=False,
                        evidence=f"checker errored: {type(e).__name__}: {e}",
                    )
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Audit directory to consume (default: latest under output/audits/)",
    )
    args = parser.parse_args()

    audit_path = args.audit_dir or latest_audit_dir()
    if audit_path is None or not audit_path.exists():
        print("ERROR: no audit directory found; run audit_surfaces.py first", file=sys.stderr)
        return 2

    print(f"Loading captures from {audit_path}")
    captures, api_status = load_captures(audit_path)
    print(f"  {len(captures)} captures loaded")

    results = run_all_checks(captures, api_status)
    failed = [r for r in results if not r.passed]
    print(f"  {len(results)} rule checks, {len(failed)} violations")

    out_path = audit_path / "rule-results.json"
    out_path.write_text(json.dumps([to_dict(r) for r in results], indent=2))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
