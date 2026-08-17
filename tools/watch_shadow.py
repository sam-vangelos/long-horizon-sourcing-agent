#!/usr/bin/env python3
"""Live tail of GLM shadow-judge activity during a run.

Follows a state dir's run_log.jsonl and pretty-prints every
facial_shadow_comparison / full_shadow_comparison event as it lands,
plus a running agreement tally. Read-only; safe to start/stop anytime.

Usage:
    python tools/watch_shadow.py output/state/linkedin/2078524586
"""
from __future__ import annotations

import functools
import json
import sys
import time

print = functools.partial(print, flush=True)
from pathlib import Path

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _fmt_single(ev: dict) -> str:
    tier = "FULL " if ev.get("event") == "full_shadow_comparison" else "facial"
    if ev.get("shadow_error"):
        return f"{DIM}[{tier}] shadow error: {str(ev.get('shadow_error'))[:80]}{RESET}"
    agrees = ev.get("agrees")
    mark = (
        f"{GREEN}AGREE{RESET}" if agrees
        else f"{RED}DISAGREE{RESET}" if agrees is False
        else f"{DIM}n/a{RESET}"
    )
    return (
        f"[{tier}] opus={ev.get('primary_decision')} "
        f"glm={ev.get('shadow_decision')} {mark} "
        f"{DIM}{ev.get('latency_ms', '?')}ms{RESET}"
    )


def main() -> int:
    state_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not state_dir:
        print("usage: watch_shadow.py <state_dir>", file=sys.stderr)
        return 2
    log = state_dir / "run_log.jsonl"
    print(f"Watching {log} (Ctrl-C to stop; the run is unaffected)")
    agree = comparable = total = 0
    pos = 0
    while True:
        if log.exists():
            with open(log) as fh:
                fh.seek(pos)
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        # Partial line mid-write — re-read it next poll.
                        break
                    pos = fh.tell()
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = ev.get("event")
                    if name == "facial_shadow_comparison" and "agrees" in ev and not isinstance(ev.get("agrees"), list):
                        total += 1
                        if ev.get("agrees") is not None and not ev.get("shadow_error"):
                            comparable += 1
                            agree += bool(ev.get("agrees"))
                        print(_fmt_single(ev))
                    elif name == "facial_shadow_comparison":
                        # batch shape: per-candidate arrays
                        agrees_list = ev.get("agrees") or []
                        prim = ev.get("primary_decisions") or []
                        shad = ev.get("shadow_decisions") or []
                        for i, a in enumerate(agrees_list):
                            total += 1
                            mark = f"{GREEN}AGREE{RESET}" if a else (f"{RED}DISAGREE{RESET}" if a is False else f"{DIM}n/a{RESET}")
                            if a is not None and not ev.get("shadow_error"):
                                comparable += 1
                                agree += bool(a)
                            p = prim[i] if i < len(prim) else "?"
                            s = shad[i] if i < len(shad) else "?"
                            print(f"[facial/batch] opus={p} glm={s} {mark}")
                    elif name == "full_shadow_comparison":
                        total += 1
                        if ev.get("agrees") is not None and not ev.get("shadow_error"):
                            comparable += 1
                            agree += bool(ev.get("agrees"))
                        print(_fmt_single(ev))
                    else:
                        continue
                    if comparable:
                        rate = agree / comparable
                        print(f"{DIM}  running agreement: {agree}/{comparable} ({rate:.1%}) over {total} comparisons{RESET}")
        time.sleep(2)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
