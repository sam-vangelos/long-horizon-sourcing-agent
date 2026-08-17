#!/usr/bin/env python3
"""Walk every Cloris surface in headless Chromium and capture facts.

Outputs to output/audits/<timestamp>/:
  - <slug>.full.png     full-page screenshot
  - <slug>.fold.png     viewport-only screenshot
  - <slug>.html         rendered DOM
  - <slug>.facts.json   per-element style + text facts
  - <slug>.meta.json    capture metadata (console / errors / network)
  - api-status.json     /api/status snapshot at capture time
  - captures.json       index of every SurfaceCapture in this run

Routes walked:
  - #/                  homescreen
  - #/filed             filed-away list
  - #/brief/new         onboarding placeholder
  - #/run/<source>/<state_key>/<run_id>  run report (5 representative variants)
  - #/totally-bogus-route  404 fallback

Run-report variants are auto-discovered from /api/status so the audit
adapts as runs are archived or new ones land.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_common import (
    DEFAULT_PORT,
    VIEWPORTS,
    ElementFact,
    SurfaceCapture,
    audit_dir,
    to_dict,
)


class HarnessAuthError(RuntimeError):
    """Raised when the harness cannot bootstrap a session token.

    Slice 53c: the harness used to call ``/api/status`` and friends with
    no Authorization header. Under strict-auth (default once
    ``CLORIS_SKIP_AUTH_FOR_TESTING`` is unset) every such call would 401
    and the harness would either crash with an obscure HTTPError or
    silently swallow the 401 in a bare ``except`` block. Both modes hid
    the real failure mode. ``HarnessAuthError`` makes the contract loud:
    ``main()`` exits non-zero with a clear message rather than producing
    a sparse, misleading audit dir.
    """


def bootstrap_session_token(base_url: str, timeout: float = 10.0) -> str:
    """Fetch a Bearer token from ``/api/bootstrap``.

    The endpoint is auth-exempt (see ``cloris/api/auth.py:_EXEMPT_EXACT``)
    so the harness can call it without already having a token. The
    response shape is ``{"token": "<opaque>", ...}``.
    """
    url = f"{base_url}/api/bootstrap"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        raise HarnessAuthError(
            f"could not reach {url}: {e}. is the backend running?"
        ) from e
    except json.JSONDecodeError as e:
        raise HarnessAuthError(
            f"{url} returned non-JSON: {e}"
        ) from e
    token = payload.get("token")
    if not token:
        raise HarnessAuthError(f"{url} returned no token: {payload!r}")
    return token


def _authenticated_get_json(
    url: str,
    token: str,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """GET ``url`` with ``Authorization: Bearer <token>`` and parse JSON.

    Used by every harness call that hits an auth-gated endpoint
    (``/api/status``, ``/api/workspace/...``, ``/api/briefs``,
    ``/api/markets``, ``/api/monitor/index``). Funneling all such calls
    through one helper means a future strict-auth tightening can't
    silently 401 a sub-discovery the way the pre-53c bare ``urlopen``
    calls did.
    """
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _redact_token(token: str) -> str:
    """Return ``token`` truncated to its first 8 chars for harness_meta logging."""
    if len(token) <= 8:
        return token
    return token[:8] + "…"


# JS extracted via page.evaluate() to capture per-element facts. Selects
# every element that does meaningful semantic / typographic work — skips
# layout-only divs unless they carry text-transform: uppercase (for R17).
EXTRACT_FACTS_JS = r"""
() => {
  const TAGS = ['h1','h2','h3','h4','button','a','dt','dd','span','p','li','label','strong','em','small','time'];
  function shortSelector(el) {
    if (el.id) return '#' + el.id;
    let s = el.tagName.toLowerCase();
    if (el.className && typeof el.className === 'string') {
      const cls = el.className.trim().split(/\s+/).filter(c => c).slice(0,2).join('.');
      if (cls) s += '.' + cls;
    }
    return s;
  }
  const out = [];
  for (const tag of TAGS) {
    for (const el of document.querySelectorAll(tag)) {
      const cs = getComputedStyle(el);
      const text = (el.textContent || '').trim();
      if (!text && tag !== 'a' && tag !== 'button') continue;
      const rect = el.getBoundingClientRect();
      const isVisible = rect.width > 0 && rect.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
      out.push({
        tag,
        text: text.slice(0, 240),
        classes: Array.from(el.classList || []),
        font_size_px: parseFloat(cs.fontSize) || 0,
        text_transform: cs.textTransform || 'none',
        role: el.getAttribute('role'),
        aria_label: el.getAttribute('aria-label'),
        selector: shortSelector(el),
        is_visible: isVisible,
      });
    }
  }
  return out;
}
"""


def _safe_slug_fragment(value: str, limit: int = 24) -> str:
    """Sanitize ``value`` for use inside a capture filename.

    Replaces filesystem-hostile characters and truncates to ``limit``.
    Used by both run-report and dynamic discovery slugs so two distinct
    ``state_key`` / ``brief_id`` / market-key inputs don't collide.
    """
    cleaned = (
        value.replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "-")
        .replace(":", "-")
        .replace("?", "")
        .replace("&", "-")
        .replace("=", "-")
    )
    return cleaned[:limit] or "unknown"


def discover_run_targets(api_status: dict[str, Any], n: int = 5) -> list[tuple[str, str, str, int, str]]:
    """Pick representative run targets from /api/status.

    Returns list of (slug, source, state_key, run_id, status_label).
    Tries to cover variety: running / interrupted / error / governor /
    completed when available.

    Slice 53b: slugs include a sanitized ``state_key`` slice so that two
    distinct ``(source, state_key)`` pairs sharing
    ``(source[:2], status[:3], run_id)`` no longer collide. Without this,
    e.g. both LinkedIn fixture entries
    (``head_of_applied_ai_fixture`` and ``senior_backend_fintech_fixture``,
    both ``status=completed``, both ``run_id=1``) overwrote each other's
    capture files.
    """
    entries = api_status.get("entries", [])
    by_status: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        lr = e.get("latest_run") or {}
        status = lr.get("status")
        if not status:
            continue
        by_status.setdefault(status, []).append(e)

    # Preference order — variety of states for thorough coverage.
    preference = [
        "running",
        "interrupted",
        "governor_limit_reached",
        "error",
        "completed",
        "succeeded",
        "abandoned",
    ]
    targets: list[tuple[str, str, str, int, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for status in preference:
        for e in by_status.get(status, []):
            key = (e["source"], e["state_key"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            state_key_safe = _safe_slug_fragment(e["state_key"])
            slug = (
                f"run-{e['source'][:2]}-{status[:3]}-"
                f"{state_key_safe}-{e['latest_run']['id']}"
            )
            targets.append(
                (
                    slug,
                    e["source"],
                    e["state_key"],
                    e["latest_run"]["id"],
                    status,
                )
            )
            if len(targets) >= n:
                return targets
    return targets


def base_routes() -> list[tuple[str, str, str]]:
    """Return [(slug, route_path, description), ...] for non-run-report routes.

    Slice 53e: covers the static surfaces called out in
    docs/full-surface-qa-playbook.md that don't require fixture lookup
    — S08 legacy intake, S10 monitor index, S15 market list, S18 tools
    shelf, S19 settings. Dynamic surfaces (S06 brief detail, S10 monitor
    run, S13 workspace-identity, S16 market detail, S17 reflection,
    RefreshBrief, workspace/candidate legacy redirects) come in via the
    discover_* helpers in main().
    """
    return [
        ("home", "/#/", "Homescreen"),
        ("filed", "/#/filed", "Filed-away list"),
        ("briefs", "/#/briefs", "Brief library (Phase D Slice D1)"),
        ("brief-new", "/#/brief/new", "Onboarding flow / Write a brief (Phase D Slice D3)"),
        ("drafts", "/#/drafts", "In-flight intake drafts (Phase D Slice D4)"),
        # Slice 53e additions (S08, S10, S15, S18, S19) — playbook routes
        # that weren't being captured at all under the prior harness.
        ("legacy-intake", "/#/brief/new?legacy=1", "Legacy intake wizard (S08)"),
        ("monitor", "/#/monitor", "Live monitor index (S10)"),
        ("market-list", "/#/market", "Market intelligence catalog (S15)"),
        ("tools", "/#/tools", "Tools shelf (S18)"),
        ("settings", "/#/settings", "Settings (S19)"),
        ("unknown", "/#/totally-bogus-route", "404 / unknown route"),
    ]


def discover_workspace_targets(
    api_status: dict[str, Any],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """Pick representative brief-first workspace targets from /api/status.

    Phase C-bis 0.1: workspace URLs are now ``#/workspace/<brief_id>``.
    Returns at most ``n`` routes; one per distinct brief_id (taken from
    each entry's ``brief_id_from_run`` field). Entries without a brief_id
    are skipped because the brief-first URL requires one.
    """
    entries = api_status.get("entries", []) or []
    seen: set[str] = set()
    targets: list[tuple[str, str, str]] = []
    for e in entries:
        brief_id = e.get("brief_id_from_run")
        if not brief_id or brief_id in seen:
            continue
        seen.add(brief_id)
        safe = _safe_slug_fragment(brief_id, limit=48)
        slug = f"workspace-{safe}"
        targets.append(
            (
                slug,
                f"/#/workspace/{brief_id}",
                f"Workspace ({brief_id})",
            )
        )
        if len(targets) >= n:
            break
    return targets


def discover_candidate_targets(
    base_url: str,
    token: str,
    api_status: dict[str, Any],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """Pick representative brief-first candidate-detail targets.

    Phase C-bis 0.1: candidate URLs are now
    ``#/candidate/<brief_id>/<candidate_id>``. For each distinct brief_id
    in /api/status, fetch the workspace (which lists saves across all
    runs of the brief) and pick the first candidate id. Briefs with no
    saves are skipped.

    Slice 53c: requires a Bearer token; previously this used a bare
    ``urllib.request.urlopen`` plus a blanket ``except Exception:
    continue`` which silently 401-ed under strict-auth and produced an
    audit with no candidate captures. The narrow catch below logs a
    visible warning so the next failure mode is the genuinely-empty
    workspace case, not auth.

    Slice 53d: the duplicate trailing ``return targets`` was unreachable
    dead code; deleted.
    """
    entries = api_status.get("entries", []) or []
    seen: set[str] = set()
    targets: list[tuple[str, str, str]] = []
    for e in entries:
        brief_id = e.get("brief_id_from_run")
        if not brief_id or brief_id in seen:
            continue
        seen.add(brief_id)
        url = f"{base_url}/api/workspace/{brief_id}"
        try:
            workspace = _authenticated_get_json(url, token)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as err:
            print(
                f"warn: candidate discovery skipped {url}: {err}",
                file=sys.stderr,
            )
            continue
        candidates = workspace.get("candidates") or []
        if not candidates:
            continue
        candidate_id = candidates[0].get("candidate_id")
        if candidate_id is None:
            continue
        safe = _safe_slug_fragment(brief_id, limit=32)
        slug = f"candidate-{safe}-{candidate_id}"
        targets.append(
            (
                slug,
                f"/#/candidate/{brief_id}/{candidate_id}",
                f"Candidate detail ({brief_id}/{candidate_id})",
            )
        )
        if len(targets) >= n:
            break
    return targets


def fetch_api_status(base_url: str, token: str) -> dict[str, Any]:
    """Snapshot ``/api/status``. Walker fails fast if backend is unreachable.

    Slice 53c: now requires a Bearer token and routes through
    ``_authenticated_get_json``. Without this, the harness 401-ed
    silently and produced an empty audit even though the backend was
    healthy.
    """
    return _authenticated_get_json(f"{base_url}/api/status", token)


# ---------------------------------------------------------------------------
# Slice 53e — dynamic discovery helpers for fixture-driven surfaces.
#
# Each helper queries an auth-gated catalog endpoint and emits at most
# ``n`` deterministic ``(slug, route, description)`` tuples. When a
# catalog returns empty (e.g. zero markets), the helper records a
# ``MissingEvidence`` entry instead of synthesizing a fake route — that
# keeps the audit honest under the playbook's "missing evidence" column.
# ---------------------------------------------------------------------------


def _discover_via_endpoint(
    base_url: str,
    token: str,
    api_path: str,
    surface: str,
    missing_evidence: list[dict[str, Any]],
) -> list[Any] | None:
    """Internal helper: GET an auth-gated list endpoint, returning ``None``
    on failure (after emitting a missing_evidence entry) or an empty list
    if the catalog is genuinely empty. Returns the list payload otherwise.
    """
    url = f"{base_url}{api_path}"
    try:
        payload = _authenticated_get_json(url, token)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"warn: {url} skipped: {e}", file=sys.stderr)
        missing_evidence.append(
            {
                "surface": surface,
                "reason": f"{api_path} unreachable: {e}",
                "needs_fixture": False,
            }
        )
        return None
    if isinstance(payload, dict):
        for key in ("entries", "items", "briefs", "markets", "tools", "runs", "results"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        # No known list field — treat as empty.
        return []
    if isinstance(payload, list):
        return payload
    return []


def discover_brief_detail_targets(
    base_url: str,
    token: str,
    missing_evidence: list[dict[str, Any]],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """S06 — brief detail for the first ``n`` discoverable briefs."""
    items = _discover_via_endpoint(
        base_url, token, "/api/briefs", "S06", missing_evidence
    )
    if items is None:
        return []
    if not items:
        missing_evidence.append(
            {
                "surface": "S06",
                "reason": "/api/briefs returned empty list",
                "needs_fixture": True,
            }
        )
        return []
    targets: list[tuple[str, str, str]] = []
    for item in items[:n]:
        brief_id = item.get("brief_id") or item.get("id")
        if not brief_id:
            continue
        safe = _safe_slug_fragment(brief_id, limit=32)
        targets.append(
            (
                f"brief-{safe}",
                f"/#/brief/{brief_id}",
                f"Brief detail ({brief_id})",
            )
        )
    return targets


def discover_monitor_run_targets(
    base_url: str,
    token: str,
    missing_evidence: list[dict[str, Any]],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """S10 — monitor-run detail for the first ``n`` active runs."""
    items = _discover_via_endpoint(
        base_url, token, "/api/monitor/index", "S10", missing_evidence
    )
    if items is None:
        return []
    if not items:
        missing_evidence.append(
            {
                "surface": "S10",
                "reason": "/api/monitor/index returned empty list",
                "needs_fixture": True,
            }
        )
        return []
    targets: list[tuple[str, str, str]] = []
    for item in items[:n]:
        source = item.get("source")
        state_key = item.get("state_key")
        run_id = item.get("run_id") or (item.get("latest_run") or {}).get("id")
        if not (source and state_key and run_id is not None):
            continue
        safe = _safe_slug_fragment(state_key, limit=24)
        targets.append(
            (
                f"monitor-run-{safe}",
                f"/#/monitor/{source}/{state_key}/{run_id}",
                f"Monitor run ({source}/{state_key}/{run_id})",
            )
        )
    return targets


def discover_workspace_identity_targets(
    api_status: dict[str, Any],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """S13 — workspace identity tab for the first ``n`` distinct briefs.

    Mirrors ``discover_workspace_targets`` so identity coverage exists
    alongside the workspace-overview capture. No additional API calls.
    """
    entries = api_status.get("entries", []) or []
    seen: set[str] = set()
    targets: list[tuple[str, str, str]] = []
    for e in entries:
        brief_id = e.get("brief_id_from_run")
        if not brief_id or brief_id in seen:
            continue
        seen.add(brief_id)
        safe = _safe_slug_fragment(brief_id, limit=48)
        targets.append(
            (
                f"workspace-identity-{safe}",
                f"/#/workspace/{brief_id}/identity",
                f"Workspace identity tab ({brief_id})",
            )
        )
        if len(targets) >= n:
            break
    return targets


def discover_market_targets(
    base_url: str,
    token: str,
    missing_evidence: list[dict[str, Any]],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """S16 — market detail for the first ``n`` discoverable markets."""
    items = _discover_via_endpoint(
        base_url, token, "/api/markets", "S16", missing_evidence
    )
    if items is None:
        return []
    if not items:
        missing_evidence.append(
            {
                "surface": "S16",
                "reason": "/api/markets returned empty list",
                "needs_fixture": True,
            }
        )
        return []
    targets: list[tuple[str, str, str]] = []
    for item in items[:n]:
        market_key = (
            item.get("key")
            or item.get("market_id")
            or item.get("slug")
            or item.get("id")
        )
        if not market_key:
            continue
        safe = _safe_slug_fragment(str(market_key), limit=32)
        targets.append(
            (
                f"market-{safe}",
                f"/#/market/{market_key}",
                f"Market detail ({market_key})",
            )
        )
    return targets


def discover_reflection_targets(
    api_status: dict[str, Any],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """S17 — reflection panel for briefs whose latest run has finished.

    The reflection surface is mounted at ``#/workspace/<brief>/reflect``
    (per F-S17-001). Only completed/succeeded briefs are pickable
    because reflection has no useful content for in-flight runs.
    """
    entries = api_status.get("entries", []) or []
    seen: set[str] = set()
    targets: list[tuple[str, str, str]] = []
    finished = {"completed", "succeeded"}
    for e in entries:
        brief_id = e.get("brief_id_from_run")
        status = (e.get("latest_run") or {}).get("status")
        if not brief_id or brief_id in seen or status not in finished:
            continue
        seen.add(brief_id)
        safe = _safe_slug_fragment(brief_id, limit=48)
        targets.append(
            (
                f"reflection-{safe}",
                f"/#/workspace/{brief_id}/reflect",
                f"Reflection ({brief_id})",
            )
        )
        if len(targets) >= n:
            break
    return targets


def discover_refresh_brief_targets(
    api_status: dict[str, Any],
    n: int = 2,
) -> list[tuple[str, str, str]]:
    """RefreshBrief — re-issue flow for the first ``n`` distinct briefs."""
    entries = api_status.get("entries", []) or []
    seen: set[str] = set()
    targets: list[tuple[str, str, str]] = []
    for e in entries:
        brief_id = e.get("brief_id_from_run")
        if not brief_id or brief_id in seen:
            continue
        seen.add(brief_id)
        safe = _safe_slug_fragment(brief_id, limit=32)
        targets.append(
            (
                f"refresh-brief-{safe}",
                f"/#/refresh-brief?brief_id={brief_id}",
                f"RefreshBrief ({brief_id})",
            )
        )
        if len(targets) >= n:
            break
    return targets


def discover_legacy_redirect_targets(
    api_status: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Legacy workspace + candidate redirect routes (cap 1 each).

    The redirect path is identical regardless of fixture, so a single
    capture per redirect is enough to verify the redirect contract
    holds. Picks the first entry from ``/api/status`` that has the
    necessary fields.
    """
    targets: list[tuple[str, str, str]] = []
    entries = api_status.get("entries", []) or []
    workspace_done = False
    candidate_done = False
    for e in entries:
        source = e.get("source")
        state_key = e.get("state_key")
        if not (source and state_key):
            continue
        if not workspace_done:
            safe = _safe_slug_fragment(state_key, limit=24)
            targets.append(
                (
                    f"workspace-legacy-{safe}",
                    f"/#/workspace/{source}/{state_key}",
                    f"Legacy workspace redirect ({source}/{state_key})",
                )
            )
            workspace_done = True
        if not candidate_done:
            safe = _safe_slug_fragment(state_key, limit=24)
            # Legacy candidate redirect uses (source, state_key) without
            # a candidate_id; the route handler should redirect to the
            # brief-first form.
            targets.append(
                (
                    f"candidate-legacy-{safe}",
                    f"/#/candidate/{source}/{state_key}",
                    f"Legacy candidate redirect ({source}/{state_key})",
                )
            )
            candidate_done = True
        if workspace_done and candidate_done:
            break
    return targets


def walk_surface(
    page,
    out_dir: Path,
    slug: str,
    route: str,
    description: str,
    base_url: str,
    viewport_w: int,
    viewport_h: int,
    install_route_handlers=None,
    splash_wait_seconds: float = 6.0,
) -> SurfaceCapture:
    """Navigate to one route at one viewport and capture everything.

    Slice 53g: ``install_route_handlers`` is an optional callable
    ``(page) -> None`` that registers Playwright ``page.route(...)``
    interceptors before navigation. Used by recovery captures to force
    deterministic error branches (bootstrap-fail, /api/* 500, network
    abort) without touching product code.
    """
    console_msgs: list[dict[str, Any]] = []
    page_errors: list[str] = []
    failed_requests: list[dict[str, Any]] = []

    page.on("console", lambda msg: console_msgs.append({"type": msg.type, "text": msg.text}))
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.on(
        "requestfailed",
        lambda req: failed_requests.append({"url": req.url, "failure": req.failure}),
    )

    url = base_url + route
    capture_slug = f"{slug}@{viewport_w}"
    cap = SurfaceCapture(
        slug=capture_slug,
        route=route,
        viewport_w=viewport_w,
        viewport_h=viewport_h,
        description=description,
        full_page_screenshot="",
        viewport_screenshot="",
        dom_html="",
        title="",
        walked_at=datetime.utcnow().isoformat() + "Z",
    )

    try:
        page.set_viewport_size({"width": viewport_w, "height": viewport_h})
        if install_route_handlers is not None:
            install_route_handlers(page)
        page.goto(url, wait_until="networkidle", timeout=30000)
        # Wait for splash to clear (5s in production frontend; SPLASH_DURATION_MS).
        time.sleep(splash_wait_seconds)
        try:
            page.wait_for_function(
                "!document.querySelector('.splash-screen')", timeout=10000
            )
        except Exception:
            pass
        time.sleep(0.5)

        full_path = out_dir / f"{capture_slug}.full.png"
        fold_path = out_dir / f"{capture_slug}.fold.png"
        dom_path = out_dir / f"{capture_slug}.html"
        facts_path = out_dir / f"{capture_slug}.facts.json"
        meta_path = out_dir / f"{capture_slug}.meta.json"

        page.screenshot(path=str(full_path), full_page=True)
        page.screenshot(path=str(fold_path), full_page=False)
        dom_path.write_text(page.content())

        raw_facts = page.evaluate(EXTRACT_FACTS_JS)
        facts = [ElementFact(**f) for f in raw_facts]
        facts_path.write_text(json.dumps([to_dict(f) for f in facts], indent=2))

        cap.full_page_screenshot = str(full_path.relative_to(out_dir))
        cap.viewport_screenshot = str(fold_path.relative_to(out_dir))
        cap.dom_html = str(dom_path.relative_to(out_dir))
        cap.facts = facts
        cap.title = page.title()
        cap.console = console_msgs
        cap.page_errors = page_errors
        cap.failed_requests = failed_requests
        meta_path.write_text(
            json.dumps(
                {
                    "console": console_msgs,
                    "page_errors": page_errors,
                    "failed_requests": failed_requests,
                    "title": cap.title,
                    "url": url,
                },
                indent=2,
            )
        )
    except Exception as e:
        cap.error = f"{type(e).__name__}: {e}"
        # Slice 53f: persist the error next to the (possibly missing)
        # capture so a downstream report-builder can show what failed
        # without walking the captures.json error field.
        try:
            (out_dir / f"{capture_slug}.error.txt").write_text(cap.error)
        except OSError:
            pass
    return cap


# ---------------------------------------------------------------------------
# Slice 53g — scripted recovery surface captures.
#
# Three deterministic error branches the harness can drive without any
# product change. Each returns a tuple ``(slug, description, handler)``
# where ``handler`` is the closure walk_surface installs onto its page
# before navigation. ErrorBoundary's runtime-throw branch can't be
# captured this way (it requires a forced throw inside a Svelte child)
# and is recorded as missing_evidence by main().
# ---------------------------------------------------------------------------


def _build_recovery_handlers() -> list[tuple[str, str, str, Any]]:
    """Return list of (slug, route, description, install_callable)."""

    def _intercept_500(url_pattern: str):
        def handler(page):
            page.route(
                url_pattern,
                lambda route: route.fulfill(
                    status=500,
                    content_type="application/json",
                    body=json.dumps({"detail": "synthetic-500-from-audit-harness"}),
                ),
            )
        return handler

    def _intercept_abort(url_pattern: str):
        def handler(page):
            page.route(url_pattern, lambda route: route.abort())
        return handler

    return [
        (
            "recovery-bootstrap-fail",
            "/#/",
            "S20 — /api/bootstrap returns 500 (welcome-gate hard fail)",
            _intercept_500("**/api/bootstrap"),
        ),
        (
            "recovery-api-500",
            "/#/",
            "S20 — /api/status returns 500 (home empty-with-error path)",
            _intercept_500("**/api/status"),
        ),
        (
            "recovery-network-down",
            "/#/",
            "S20 — every /api/* aborts (offline copy)",
            _intercept_abort("**/api/**"),
        ),
    ]


def _device_scale_factor_for(viewport_w: int) -> int:
    """Slice 53f: render every viewport at DSF=1 to keep Chromium alive.

    The plan called for DSF=1 only at >=1440. Empirically, after the
    Slice 53e route expansion, even the 1024 sweep at DSF=2 trips the
    same ``GPU process isn't usable`` crash within the first 5 captures
    — Chromium's tile-memory limit triggers on every page. Audit
    captures don't need retina (they're consumed by ``audit_rules.py``
    + downstream ensemble review, not by humans on retina displays);
    the failure mode of "no captures at all" is strictly worse than
    "captures at 1x".

    The signature stays per-viewport so a future env that can sustain
    DSF=2 at 1024 only can revert this without ripping out the wiring.
    """
    return 1


def _build_routes(
    base_url: str,
    token: str,
    api_status: dict[str, Any],
    missing_evidence: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Assemble the full route list from base + dynamic discovery helpers.

    Slice 53e: layered coverage. The ``base_routes()`` block covers
    static surfaces (S04, S07, S08, S10 index, S15 list, S18, S19, 404).
    Dynamic helpers below cover S06, S10 run, S13 identity, S16, S17,
    RefreshBrief, and the legacy redirects. Helpers append to
    ``missing_evidence`` whenever a fixture-driven surface has no
    discoverable target so the audit is honest about coverage gaps.
    """
    routes: list[tuple[str, str, str]] = list(base_routes())

    run_targets = discover_run_targets(api_status, n=5)
    for slug, source, state_key, run_id, status in run_targets:
        routes.append(
            (slug, f"/#/run/{source}/{state_key}/{run_id}", f"Run report ({status})")
        )

    # Phase C-bis 0.1 surfaces — brief-first workspace and candidate detail.
    routes.extend(discover_workspace_targets(api_status, n=2))
    routes.extend(discover_candidate_targets(base_url, token, api_status, n=2))

    # Slice 53e additions.
    routes.extend(discover_brief_detail_targets(base_url, token, missing_evidence, n=2))
    routes.extend(discover_monitor_run_targets(base_url, token, missing_evidence, n=2))
    routes.extend(discover_workspace_identity_targets(api_status, n=2))
    routes.extend(discover_market_targets(base_url, token, missing_evidence, n=2))
    routes.extend(discover_reflection_targets(api_status, n=2))
    routes.extend(discover_refresh_brief_targets(api_status, n=2))
    routes.extend(discover_legacy_redirect_targets(api_status))

    return routes


def _record_known_missing_evidence(
    api_status: dict[str, Any],
    missing_evidence: list[dict[str, Any]],
) -> None:
    """Pre-register the surfaces the harness can't capture without product hooks.

    Slice 53e: S09 sub-states (readiness blocker, multi-module preview)
    and S11 live-signal expanded variants are interactive UI states that
    don't have URL routes. Slice 53g: ErrorBoundary's runtime-throw
    branch needs a product-side throw hook.

    Recording these here keeps the audit honest under the playbook's
    "missing evidence" column rather than leaving gaps unspoken.
    """
    home_capture = "home@<viewport>"
    missing_evidence.extend(
        [
            {
                "surface": "S09",
                "branch": "readiness-blocker",
                "reason": "interactive launch-form sub-state; no URL route",
                "needs_product_change": "expose dev affordance to force-render readiness blocker",
                "see_capture": home_capture,
            },
            {
                "surface": "S09",
                "branch": "multi-module-preview",
                "reason": "interactive launch-form sub-state; no URL route",
                "needs_product_change": "expose dev affordance to force-render multi-module preview",
                "see_capture": home_capture,
            },
            {
                "surface": "S11",
                "branch": "live-signal-expanded-variants",
                "reason": "expansion is interactive; the unexpanded path is captured via home",
                "needs_product_change": "consider URL-state for expanded panels for testability",
                "see_capture": home_capture,
            },
            {
                "surface": "S20",
                "branch": "errorboundary-render",
                "reason": "no deterministic throw hook in App.svelte",
                "needs_product_change": "add ?__forceErrorBoundary=1 dev hook in App.svelte (out of scope for Slice 53)",
            },
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CLORIS_PORT", DEFAULT_PORT)),
        help=f"Cloris backend port (default {DEFAULT_PORT}, or $CLORIS_PORT)",
    )
    parser.add_argument(
        "--viewports",
        nargs="+",
        type=int,
        default=[1280],
        help="Viewport widths to walk (default: 1280; pass multiple for full sweep)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: output/audits/<timestamp>)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append captures to an existing output dir (Slice 53f Makefile-split "
            "fallback). Requires --out."
        ),
    )
    args = parser.parse_args()

    if args.append and args.out is None:
        print("ERROR: --append requires --out", file=sys.stderr)
        return 4

    base_url = f"http://127.0.0.1:{args.port}"

    # Slice 53c: bootstrap the session token before any other API call.
    try:
        token = bootstrap_session_token(base_url)
    except HarnessAuthError as e:
        print(f"ERROR: harness auth bootstrap failed: {e}", file=sys.stderr)
        print(
            "       run `python -m cloris start` (or set --port / CLORIS_PORT)",
            file=sys.stderr,
        )
        return 2

    try:
        api_status = fetch_api_status(base_url, token)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: /api/status unreachable: {e}", file=sys.stderr)
        return 2

    out_dir = args.out or audit_dir()
    print(f"Audit output: {out_dir}")

    # Slice 53f append mode: don't clobber api-status.json / harness_meta
    # / api-bootstrap on subsequent invocations.
    if not args.append:
        (out_dir / "api-status.json").write_text(json.dumps(api_status, indent=2))
        (out_dir / "harness_meta.json").write_text(
            json.dumps(
                {
                    "started_at": datetime.utcnow().isoformat() + "Z",
                    "base_url": base_url,
                    "viewports": args.viewports,
                    "session_token_prefix": _redact_token(token),
                    "strict_auth": (
                        os.environ.get("CLORIS_SKIP_AUTH_FOR_TESTING") not in ("1", "true", "True")
                    ),
                },
                indent=2,
            )
        )

    missing_evidence: list[dict[str, Any]] = []
    _record_known_missing_evidence(api_status, missing_evidence)
    routes = _build_routes(base_url, token, api_status, missing_evidence)
    recovery_targets = _build_recovery_handlers()

    captures: list[SurfaceCapture] = []
    if args.append:
        existing = out_dir / "captures.json"
        if existing.exists():
            try:
                # Reload prior captures so the merged file remains the
                # canonical index across split invocations. Re-instantiating
                # SurfaceCapture would require ElementFact rehydration; the
                # walker only consumes the merged JSON via to_dict, so we
                # re-emit the prior dicts directly below.
                pass
            except Exception:
                pass

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed (pip install playwright)", file=sys.stderr)
        return 3

    # Slice 53f: launch one browser per viewport so GPU/memory pressure
    # doesn't accumulate across the full sweep. Combined with DSF=1 at
    # >=1440 and the extra Chromium args, this keeps the 1440 walk from
    # crashing mid-sweep the way the prior single-browser shape did.
    chromium_args = [
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-software-rasterizer",
        "--disable-features=VizDisplayCompositor",
    ]

    with sync_playwright() as pw:
        for viewport_w in args.viewports:
            viewport_h = 900
            dsf = _device_scale_factor_for(viewport_w)
            print(f"viewport sweep: {viewport_w}x{viewport_h} (DSF={dsf})")
            browser = pw.chromium.launch(headless=True, args=chromium_args)
            ctx = browser.new_context(device_scale_factor=dsf)
            try:
                # Static + dynamic surfaces.
                for slug, route, desc in routes:
                    page = ctx.new_page()
                    print(f"  walking {slug}@{viewport_w} -> {route}")
                    cap = walk_surface(
                        page=page,
                        out_dir=out_dir,
                        slug=slug,
                        route=route,
                        description=desc,
                        base_url=base_url,
                        viewport_w=viewport_w,
                        viewport_h=viewport_h,
                    )
                    captures.append(cap)
                    if cap.error:
                        print(f"    ERROR: {cap.error}")
                    page.close()

                # Slice 53g: scripted recovery captures last so a flaky
                # interceptor crash doesn't poison the rest of the sweep.
                for slug, route, desc, install_handlers in recovery_targets:
                    page = ctx.new_page()
                    print(f"  walking {slug}@{viewport_w} -> {route} (recovery)")
                    cap = walk_surface(
                        page=page,
                        out_dir=out_dir,
                        slug=slug,
                        route=route,
                        description=desc,
                        base_url=base_url,
                        viewport_w=viewport_w,
                        viewport_h=viewport_h,
                        install_route_handlers=install_handlers,
                        # Recovery branches deliberately don't reach the
                        # bootstrap-success path, so the 6s splash wait
                        # is wasted; trim to keep the audit fast.
                        splash_wait_seconds=2.0,
                    )
                    captures.append(cap)
                    if cap.error:
                        print(f"    ERROR: {cap.error}")
                    page.close()
            finally:
                ctx.close()
                browser.close()

    # Slice 53f append-mode merge: load any pre-existing captures.json
    # and append new captures (newer entries override same-slug older
    # entries; never the reverse). Keeps the file canonical across a
    # split-by-viewport invocation chain.
    captures_dicts: list[dict[str, Any]] = [to_dict(c) for c in captures]
    if args.append:
        existing_path = out_dir / "captures.json"
        if existing_path.exists():
            try:
                prior = json.loads(existing_path.read_text())
                if isinstance(prior, list):
                    seen_slugs = {c.get("slug") for c in captures_dicts}
                    captures_dicts = [
                        p for p in prior if p.get("slug") not in seen_slugs
                    ] + captures_dicts
            except json.JSONDecodeError as e:
                print(f"warn: prior captures.json unreadable: {e}", file=sys.stderr)

    (out_dir / "captures.json").write_text(json.dumps(captures_dicts, indent=2))

    # Slice 53e: persist missing_evidence so audit_report.py and human
    # reviewers can render the gap row alongside the surface table.
    missing_path = out_dir / "missing_evidence.json"
    if args.append and missing_path.exists():
        try:
            prior = json.loads(missing_path.read_text())
            if isinstance(prior, list):
                missing_evidence = prior + missing_evidence
        except json.JSONDecodeError:
            pass
    missing_path.write_text(json.dumps(missing_evidence, indent=2))

    print(f"DONE — {len(captures)} captures written to {out_dir}")
    if missing_evidence:
        print(
            f"     — {len(missing_evidence)} missing-evidence entries recorded"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
