#!/usr/bin/env python3
"""Audit .env against what the code actually reads. Prints VARIABLE NAMES ONLY.

Run from the repo root:  python3 tools/env_audit.py

For every KEY=... line in .env, searches tracked source for a literal
reference to the name and buckets the result:

  UNREFERENCED  — no hit anywhere in tracked code: safe to delete.
  TEST-ONLY     — hits only under tests/: dead for runtime, deletable.
  CONSTRUCTED   — agent-prefixed API keys (LINKEDIN_*/GITHUB_* + a base key
                  read via config._agent_key): built at runtime, no literal
                  hit expected. Live if that module runs.
  LIVE          — read by runtime code (top referencing files shown).

Values are never read into the report — only names. The one judgment the
script cannot make: a LIVE var whose .env value merely repeats the config
default is also deletable, but comparing values is deliberately out of scope.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV_PATH = REPO / ".env"

# Base keys read through config._agent_key(); AGENT_KEY_PREFIX ("LINKEDIN" /
# "GITHUB") constructs prefixed variants at runtime, so no literal hit exists.
AGENT_KEY_BASES = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "PERPLEXITY_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_M3_API_KEY",
}
AGENT_PREFIXES = ("LINKEDIN_", "GITHUB_")


def env_names() -> list[str]:
    names = []
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line)
        if m:
            names.append(m.group(1))
    return names


def grep_files(name: str) -> list[str]:
    """Tracked files containing the literal name, .env and docs excluded."""
    proc = subprocess.run(
        ["git", "grep", "-l", "-F", name, "--",
         "*.py", "*.sh", "*.toml", "Makefile"],
        capture_output=True, text=True, cwd=REPO,
    )
    return [
        f for f in proc.stdout.splitlines()
        if f and f != "tools/env_audit.py"
    ]


def main() -> int:
    if not ENV_PATH.exists():
        print("no .env at repo root", file=sys.stderr)
        return 1
    buckets: dict[str, list[tuple[str, list[str]]]] = {
        "LIVE": [], "TEST-ONLY": [], "CONSTRUCTED": [], "UNREFERENCED": [],
    }
    for name in env_names():
        hits = grep_files(name)
        if hits:
            runtime = [f for f in hits if not f.startswith("tests/")]
            if runtime:
                buckets["LIVE"].append((name, runtime[:3]))
            else:
                buckets["TEST-ONLY"].append((name, hits[:3]))
        elif any(
            name.startswith(p) and name[len(p):] in AGENT_KEY_BASES
            for p in AGENT_PREFIXES
        ):
            buckets["CONSTRUCTED"].append((name, []))
        else:
            buckets["UNREFERENCED"].append((name, []))

    for bucket in ("UNREFERENCED", "TEST-ONLY", "CONSTRUCTED", "LIVE"):
        rows = buckets[bucket]
        print(f"\n{bucket} ({len(rows)})")
        for name, files in sorted(rows):
            suffix = f"  <- {', '.join(files)}" if files else ""
            print(f"  {name}{suffix}")

    dead = len(buckets["UNREFERENCED"]) + len(buckets["TEST-ONLY"])
    print(f"\nsafe to delete: {dead} "
          "(every UNREFERENCED and TEST-ONLY line)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
