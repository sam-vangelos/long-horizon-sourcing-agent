"""Built frontend static-file routes and mounts."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles

from . import _paths
from .routing import router

log = logging.getLogger("cloris.api")

_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoStoreStaticFiles(StaticFiles):
    """StaticFiles variant for local app assets that must not survive rebuilds."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(_NO_STORE_HEADERS)
        return response


def _warn_if_dist_stale() -> None:
    """Log a warning if ``cloris/frontend/dist`` is older than source."""

    dist_index = _paths._DIST_DIR / "index.html"
    if not dist_index.is_file():
        return

    source_roots: list[tuple[Path, set[str] | None]] = []
    if _paths._FRONTEND_SRC_DIR.is_dir():
        source_roots.append((_paths._FRONTEND_SRC_DIR, None))
    if _paths._FRONTEND_SCAFFOLDS_DIR.is_dir():
        source_roots.append((_paths._FRONTEND_SCAFFOLDS_DIR, {".html"}))
    if not source_roots:
        return

    try:
        dist_mtime = dist_index.stat().st_mtime
    except OSError as exc:  # pragma: no cover - defensive
        log.debug("dist staleness check skipped: %s", exc)
        return

    newest_src_mtime = 0.0
    newest_src_path: Path | None = None
    try:
        for source_root, allowed_suffixes in source_roots:
            for src_path in source_root.rglob("*"):
                if not src_path.is_file():
                    continue
                if any(
                    part in {"node_modules", "__pycache__", ".svelte-kit", "Archive"}
                    for part in src_path.parts
                ):
                    continue
                if allowed_suffixes is not None and src_path.suffix not in allowed_suffixes:
                    continue
                try:
                    src_mtime = src_path.stat().st_mtime
                except OSError:
                    continue
                if src_mtime > newest_src_mtime:
                    newest_src_mtime = src_mtime
                    newest_src_path = src_path
    except OSError as exc:  # pragma: no cover - defensive
        log.debug("dist staleness check traversal skipped: %s", exc)
        return

    if newest_src_path is not None and newest_src_mtime > dist_mtime:
        log.warning(
            "frontend dist/ is older than src/ or scaffolds/. The browser may serve "
            "stale UI. Run `pnpm --filter cloris-frontend build` (or "
            "`make dev` in development) to rebuild. Reference: "
            "%s (mtime %d) vs. %s (mtime %d).",
            dist_index,
            int(dist_mtime),
            newest_src_path,
            int(newest_src_mtime),
        )


def mount_static(app) -> None:
    """Mount the built Cloris UI's static asset trees.

    Two prefixes are mounted from the built ``dist/`` tree:

    - ``/assets/`` — Vite's hashed JS/CSS bundle. Always present in a
      shipped build.
    - ``/brand/`` — brand icons + favicons referenced by ``index.html``
      (``<link rel="icon">``, ``<link rel="apple-touch-icon">``,
      ``og:image``). Browsers fetch these without a Bearer header, so
      the prefix is also exempt from :class:`BearerAuthMiddleware`
      (see ``cloris/api/auth.py``). Guarded with ``.is_dir()`` so a
      build that pre-dates the brand assets doesn't crash boot.

    The ``/manifest.webmanifest`` exact-match route is declared
    separately below.
    """

    _warn_if_dist_stale()
    assets_dir = _paths._DIST_DIR / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            NoStoreStaticFiles(directory=assets_dir),
            name="cloris-assets",
        )
    else:
        log.warning(
            "frontend dist assets missing at %s; serving API without the UI "
            "(the shell is parked — see attic/README.md)",
            assets_dir,
        )

    brand_dir = _paths._DIST_DIR / "brand"
    if brand_dir.is_dir():
        app.mount(
            "/brand",
            NoStoreStaticFiles(directory=brand_dir),
            name="cloris-brand",
        )
    else:
        log.warning(
            "frontend dist/brand/ is missing; favicon and brand icons will 404. "
            "Run `pnpm --filter cloris-frontend build` to populate."
        )


@router.get("/")
def index() -> FileResponse:
    """Serve the built Cloris UI's ``index.html`` (404 when the shell is parked)."""

    index_path = _paths._DIST_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="the desktop shell is parked (see attic/README.md); "
            "check out the desktop-shell-last-green tag to run the UI",
        )
    return FileResponse(
        index_path,
        media_type="text/html",
        headers=_NO_STORE_HEADERS,
    )


@router.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    """Serve the built PWA manifest referenced by ``index.html``.

    Browsers request ``/manifest.webmanifest`` (``<link rel="manifest">``)
    without an Authorization header, so the path is exempt from
    :class:`BearerAuthMiddleware`.
    """

    return FileResponse(
        _paths._DIST_DIR / "manifest.webmanifest",
        media_type="application/manifest+json",
        headers=_NO_STORE_HEADERS,
    )
