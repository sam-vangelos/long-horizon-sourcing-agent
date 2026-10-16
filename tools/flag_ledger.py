#!/usr/bin/env python3
"""Every dark feature flag names the issue and the date that will decide it.

A flag born default-off with no owner is how the facet-clearing switch sat
wrong for four weeks: nothing said what would make it safe to turn on, so
nobody turned it on. This ledger refuses that shape. A default-off flag no
launcher enables must carry, in its config comment block, either

    # gate: CLO-NNN by YYYY-MM-DD      a Linear issue and a decide-by date
    # gate: by-design — <reason>       a deliberate switch, no date owed

and a dated gate that has come due is an offender in its own right. A flag
nothing in production reads is an offender under either default.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "shared" / "config.py"
READER_AREAS = ("linkedin", "shared", "cloris", "market_intelligence")

DEFINITION = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9_]*)\s*:\s*bool\s*=\s*_env_flag\(\s*"
    r"\"(?P=name)\"\s*,\s*\"(?P<default>[a-z]+)\"\s*,?\s*\)",
    re.MULTILINE,
)
DATED_GATE = re.compile(r"#\s*gate:\s*([A-Z][A-Z0-9]*-\d+)\s+by\s+(\d{4}-\d{2}-\d{2})")
BY_DESIGN_GATE = re.compile(r"#\s*gate:\s*by-design\s*[—-]\s*(\S.*)")


class Flag:
    def __init__(self, name: str, default: str, comment: str, line: int) -> None:
        self.name = name
        self.default_on = default == "true"
        self.comment = comment
        self.line = line
        self.launchers: list[str] = []
        self.readers = 0

    @property
    def gate(self) -> str:
        dated = DATED_GATE.search(self.comment)
        if dated:
            return f"{dated[1]} by {dated[2]}"
        by_design = BY_DESIGN_GATE.search(self.comment)
        return f"by-design — {by_design[1].strip()}" if by_design else ""


def parse_config(source: str) -> list[Flag]:
    """Name, default, and the contiguous comment block above each _env_flag."""
    lines = source.splitlines()
    flags = []
    for match in DEFINITION.finditer(source):
        index = source[: match.start()].count("\n")  # 0-based line of the def
        start = index
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        comment = "\n".join(lines[start:index])
        flags.append(Flag(match["name"], match["default"], comment, index + 1))
    return flags


def launcher_exports(flags: list[Flag], launchers: list[Path]) -> None:
    for path in launchers:
        text = path.read_text()
        for flag in flags:
            if re.search(rf"^\s*export {flag.name}=true\b", text, re.MULTILINE):
                flag.launchers.append(path.name)


def count_readers(flags: list[Flag], root: Path) -> None:
    sources = []
    for area in READER_AREAS:
        for path in (root / area).rglob("*.py"):
            if path != root / "shared" / "config.py":
                sources.append(path.read_text())
    for flag in flags:
        flag.readers = sum(source.count(flag.name) for source in sources)


def offenders(flags: list[Flag], today: dt.date) -> list[str]:
    found = []
    for flag in flags:
        if not flag.readers:
            found.append(f"{flag.name}: dead flag — no production code reads it")
            continue
        if flag.default_on or flag.launchers:
            continue
        dated = DATED_GATE.search(flag.comment)
        if dated:
            if dt.date.fromisoformat(dated[2]) < today:
                found.append(
                    f"{flag.name}: gate expired {dated[2]} ({dated[1]}) — "
                    "re-decide or extend"
                )
        elif not BY_DESIGN_GATE.search(flag.comment):
            found.append(
                f"{flag.name}: default off, no launcher enables it, and its "
                "comment names no gate (# gate: CLO-NNN by YYYY-MM-DD, or "
                "# gate: by-design — <reason>)"
            )
    return found


def build(root: Path = ROOT) -> list[Flag]:
    flags = parse_config((root / "shared" / "config.py").read_text())
    launcher_exports(flags, sorted((root / "tools").glob("launch_*.sh")))
    count_readers(flags, root)
    return flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--list", action="store_true", help="print the whole table")
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=None,
        help="judge dated gates as of this YYYY-MM-DD instead of today",
    )
    args = parser.parse_args(argv)

    flags = build(args.root)
    if args.list:
        width = max(len(flag.name) for flag in flags)
        for flag in sorted(flags, key=lambda f: f.name):
            print(
                f"{flag.name:<{width}}  {'on ' if flag.default_on else 'off'}  "
                f"{', '.join(flag.launchers) or '(no launcher)':<60}  "
                f"{flag.gate or '(no gate)'}"
            )
    found = offenders(flags, args.as_of or dt.date.today())
    for offender in found:
        print(f"[flag-ledger] {offender}")
    if found:
        return 1
    dark = sum(not flag.default_on for flag in flags)
    print(f"flag ledger: {len(flags)} flags, {dark} default-off, 0 offenders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
