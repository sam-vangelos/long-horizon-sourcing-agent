#!/usr/bin/env python3
"""Lightweight repo hygiene checks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.brief_lifecycle import summarize_brief_lifecycle


def is_incidental_tracked_path(path: str) -> bool:
    """Return True when a tracked path looks like accidental repo junk."""
    parts = Path(path).parts
    name = parts[-1] if parts else path
    if name == ".DS_Store":
        return True
    if "__pycache__" in parts:
        return True
    if name.endswith(".json") and ".bak-" in name:
        return True
    return False


def tracked_incidental_files(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Return tracked files that should not normally live in the repo."""
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=project_root,
        text=True,
    )
    tracked_paths = [line.strip() for line in output.splitlines() if line.strip()]
    return sorted(path for path in tracked_paths if is_incidental_tracked_path(path))


PROVENANCE_RE = __import__("re").compile(
    r"<!-- provenance: status=(canonical|historical|superseded|aspirational|"
    r"unverified-generated) era=[a-z-]+ stamped=\d{4}-\d{2}-\d{2}"
)
PROVENANCE_EXEMPT = {"CLAUDE.md", "AGENTS.md"}
_PROV_EXCLUDED_PREFIXES = (
    "plans/", "tests/", "attic/", "docs/archive/", "cloris/frontend/",
)


def _in_provenance_scope(path: str) -> bool:
    """Tracked markdown that must carry a provenance stamp (era-alignment B4)."""
    if not path.endswith(".md"):
        return False
    if path in PROVENANCE_EXEMPT:
        return False
    if path.startswith(_PROV_EXCLUDED_PREFIXES):
        return False
    if "/jd-" in path or "node_modules" in path:
        return False
    return True


def doc_provenance_failures(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Tracked docs missing a provenance stamp in their first or last lines."""
    output = subprocess.check_output(["git", "ls-files", "*.md"], cwd=project_root, text=True)
    failures = []
    for path in (line.strip() for line in output.splitlines() if line.strip()):
        if not _in_provenance_scope(path):
            continue
        try:
            text = (project_root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            failures.append(path)
            continue
        lines = text.splitlines()
        head_tail = lines[:1] + lines[-3:]
        if not any(PROVENANCE_RE.search(line) for line in head_tail):
            failures.append(path)
    return sorted(failures)


def main() -> int:
    junk = tracked_incidental_files(PROJECT_ROOT)
    summary = summarize_brief_lifecycle(CONFIG_DIR, recursive=True)

    print("[hygiene] brief inventory")
    for lifecycle, paths in summary.items():
        print(f"  - {lifecycle}: {len(paths)}")

    failed = False
    if junk:
        print("[hygiene] tracked incidental files found:")
        for path in junk:
            print(f"  - {path}")
        failed = True
    else:
        print("[hygiene] tracked incidental files: none")

    unstamped = doc_provenance_failures(PROJECT_ROOT)
    if unstamped:
        print("[hygiene] docs missing a provenance stamp (see docs/INDEX.md):")
        for path in unstamped:
            print(f"  - {path}")
        failed = True
    else:
        print("[hygiene] doc provenance stamps: all present")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
