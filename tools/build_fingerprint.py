#!/usr/bin/env python3
"""Content-hash fingerprint of bundled-source Python files.

Used by the packaged-app staleness check
(``tools/check_packaged_modules.py``) to prove ``dist/Cloris.app``
reflects the **current** source tree, not just that the named modules
import. Embedded as a tiny text file at
``cloris/packaging/_build_revision.txt`` by
``cloris/packaging/scripts/build-app.sh`` before PyInstaller runs;
asserted at packaged-cert time by spawning the bundle's entry binary
with ``CLORIS_VERIFY_BUILD_REVISION=<expected>``.

Why a content hash rather than git HEAD: developers iterate on dirty
trees. Git SHA would falsely pass for any uncommitted change that
hasn't been bundled. The content hash captures whatever the build
saw — committed or not — and changes any time a bundled Python file
on disk changes.

Scope: only Python source under the directories PyInstaller actually
bundles into the main UI binary's PYZ archive (``cloris/``, ``shared/``,
``market_intelligence/``). Frontend assets are governed by the
existing dist-staleness guard at ``cloris/api/_paths.py``; this
fingerprint deliberately stays Python-only so it doesn't false-positive
on a frontend rebuild that the bundled binary doesn't care about.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable


# Directories whose Python source is bundled into the main UI binary
# (see ``cloris/packaging/cloris.spec``). The worker binary has its
# own dep tree but writes synthesis state through the main API, so
# any drift in these roots is what we need to detect.
BUNDLED_SOURCE_ROOTS: tuple[str, ...] = (
    "cloris",
    "shared",
    "market_intelligence",
)


# Directory + file patterns to skip when walking source roots.
_SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        # Generated test/scratch trees inside cloris/ that exist in
        # some checkouts and would otherwise destabilize the hash.
        "node_modules",
        # Bundled vite output is tracked under cloris/frontend/dist/
        # but it's not Python source.
        "dist",
    }
)
_SKIP_FILE_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd"})


def iter_bundled_python_files(repo_root: Path) -> Iterable[Path]:
    """Yield every Python source file PyInstaller bundles, in stable order."""

    for root in BUNDLED_SOURCE_ROOTS:
        base = repo_root / root
        if not base.is_dir():
            continue
        # ``sorted(rglob)`` is stable across filesystems and Python
        # versions, which is what we need for a deterministic fingerprint.
        for path in sorted(base.rglob("*.py")):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.suffix in _SKIP_FILE_SUFFIXES:
                continue
            yield path


def compute_fingerprint(repo_root: Path) -> str:
    """Return a SHA-256 hexdigest of bundled-source Python content.

    Each contributing file is fed as ``<relative_path>\\0<sha256(content)>\\0``
    so a rename moves the hash even if file content is unchanged. The
    top-level hash hashes the per-file hashes — same content, same
    fingerprint, irrespective of build host.
    """

    h = hashlib.sha256()
    for path in iter_bundled_python_files(repo_root):
        rel = path.relative_to(repo_root).as_posix().encode("utf-8")
        content_hash = hashlib.sha256(path.read_bytes()).digest()
        h.update(rel)
        h.update(b"\0")
        h.update(content_hash)
        h.update(b"\0")
    return h.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: derived from this script's location).",
    )
    args = parser.parse_args(argv)
    root = (args.root or _repo_root()).resolve()
    sys.stdout.write(compute_fingerprint(root) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
