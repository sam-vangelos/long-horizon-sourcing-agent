#!/usr/bin/env python3
"""Validate Cloris frontend static asset and package contracts."""

from __future__ import annotations

import argparse
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


REQUIRED_BRAND_FILES = (
    "cloris-icon.svg",
    "icon-16.png",
    "icon-32.png",
    "icon-180.png",
    "icon-192.png",
    "icon-512.png",
    "icon-1024.png",
)


class _AssetReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attr_map = {k: v for k, v in attrs}
        for key in ("href", "src", "content"):
            value = attr_map.get(key)
            if isinstance(value, str) and value.startswith(("/assets/", "/brand/")):
                self.refs.append(value)
        if tag == "link" and attr_map.get("rel") == "manifest":
            href = attr_map.get("href")
            if isinstance(href, str):
                self.refs.append(href)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _existing_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")


def _existing_dir(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"missing {label}: {path}")


def _resolve_dist_ref(dist: Path, ref: str) -> Path | None:
    if ref.startswith("/assets/"):
        return dist / ref.removeprefix("/")
    if ref.startswith("/brand/"):
        return dist / ref.removeprefix("/")
    if ref == "/manifest.webmanifest":
        return dist / "manifest.webmanifest"
    return None


def _manifest_refs(manifest_path: Path, errors: list[str]) -> list[str]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        errors.append(f"invalid manifest JSON: {manifest_path}: {exc}")
        return []
    icons = manifest.get("icons")
    if not isinstance(icons, list):
        errors.append(f"manifest icons must be a list: {manifest_path}")
        return []
    refs: list[str] = []
    for icon in icons:
        if not isinstance(icon, dict):
            errors.append(f"manifest icon must be an object: {manifest_path}")
            continue
        src = icon.get("src")
        if isinstance(src, str) and src.startswith("/brand/"):
            refs.append(src)
    return refs


def validate_dist(dist: Path) -> list[str]:
    """Return contract errors for a built frontend dist directory."""

    dist = dist.resolve()
    errors: list[str] = []
    _existing_dir(dist, "frontend dist", errors)
    index = dist / "index.html"
    assets = dist / "assets"
    brand = dist / "brand"
    manifest = dist / "manifest.webmanifest"

    _existing_file(index, "dist index", errors)
    _existing_dir(assets, "dist assets directory", errors)
    _existing_dir(brand, "dist brand directory", errors)
    _existing_file(manifest, "web manifest", errors)

    if brand.is_dir():
        for filename in REQUIRED_BRAND_FILES:
            _existing_file(brand / filename, f"brand asset {filename}", errors)

    refs: list[str] = []
    if index.is_file():
        parser = _AssetReferenceParser()
        parser.feed(index.read_text(errors="replace"))
        refs.extend(parser.refs)

    if assets.is_dir():
        if not any(assets.glob("*.js")):
            errors.append(f"missing built JS asset under {assets}")
        if not any(assets.glob("*.css")):
            errors.append(f"missing built CSS asset under {assets}")
        referenced_assets = {
            Path(ref.removeprefix("/assets/")).name
            for ref in refs
            if ref.startswith("/assets/")
        }
        for pattern in ("index-*.js", "index-*.css"):
            for asset in sorted(assets.glob(pattern)):
                if asset.name not in referenced_assets:
                    errors.append(
                        f"stale Vite entry asset {asset} is not referenced by index.html"
                    )

    refs.extend(_manifest_refs(manifest, errors))
    for ref in sorted(set(refs)):
        target = _resolve_dist_ref(dist, ref)
        if target is not None and not target.is_file():
            errors.append(f"unresolved static reference {ref}: expected {target}")

    return errors


def validate_app(app: Path) -> list[str]:
    """Return smoke-contract errors for a local Cloris.app bundle."""

    app = app.resolve()
    errors: list[str] = []
    _existing_dir(app, "Cloris.app", errors)
    main_binary = app / "Contents" / "MacOS" / "Cloris"
    worker_binary = app / "Contents" / "MacOS" / "cloris-worker"
    _existing_file(main_binary, "main app binary", errors)
    _existing_file(worker_binary, "worker binary", errors)
    for binary in (main_binary, worker_binary):
        if binary.exists() and not binary.stat().st_mode & 0o111:
            errors.append(f"binary is not executable: {binary}")

    bundled_dist = app / "Contents" / "Resources" / "cloris" / "frontend" / "dist"
    errors.extend(validate_dist(bundled_dist))
    return errors


def _print_errors(errors: Iterable[str]) -> None:
    for error in errors:
        print(f"[static-assets] {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        type=Path,
        default=None,
        help="Validate a frontend dist directory.",
    )
    parser.add_argument(
        "--app",
        type=Path,
        default=None,
        help="Validate a Cloris.app bundle and its bundled frontend dist.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    if args.dist is None and args.app is None:
        args.dist = _repo_root() / "cloris" / "frontend" / "dist"
    if args.dist is not None:
        errors.extend(validate_dist(args.dist))
    if args.app is not None:
        errors.extend(validate_app(args.app))

    if errors:
        _print_errors(errors)
        return 1
    print("[static-assets] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
