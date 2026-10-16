#!/usr/bin/env python3
"""Rerun the enricher sidecar over a stored plan, and write nothing.

A live formation shows what the sidecar did once, on whatever string the
strategy model wrote that day. This replays the same per-string logic the
formation hook runs (``linkedin.boolean_sidecar._improve``) against a real
sidecar, on every string of a plan already on disk, as authored: a string a
pass already rewrote is replayed from its ``_original``. The flag is not
consulted, the plan is only read, and nothing is written anywhere.

For each string it prints what the string offered (single-concept groups
sent whole, concept runs inside mixed groups, and how many of those the
per-string cap lets through), the requests made, every change kept with the
string and its score before and after, and the wall time. Then the totals.

    .venv/bin/python tools/sidecar_replay.py output/state/linkedin/<project>
    .venv/bin/python tools/sidecar_replay.py <state-dir> --url http://127.0.0.1:8765 --limit 5

Exit 0 when every string was offered, 1 when the sidecar stopped answering
part way (the strings after it are not replayed), 2 when the plan or the
lexicon cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shared.config as _config  # noqa: E402
from linkedin import boolean_sidecar  # noqa: E402
from linkedin.boolean_enrichment import load_role_lexicon  # noqa: E402
from linkedin.boolean_score.lexicon import Lexicon  # noqa: E402
from linkedin.boolean_score.parser import parse  # noqa: E402

PLAN_FILE = "execution_plan.json"
# The plan lists and fields the formation hook offers to the sidecar.
SURFACES = (("generated_strings", "boolean"), ("coverage_gaps", "suggested_boolean"))


def plan_strings(plan: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """``(label, boolean)`` for every string in the plan, as authored."""
    for key, field in SURFACES:
        for index, item in enumerate(plan.get(key) or []):
            if not isinstance(item, dict):
                continue
            original = item.get(f"{field}_original")
            boolean = original if isinstance(original, str) and original.strip() else item.get(field)
            if isinstance(boolean, str) and boolean.strip():
                yield f"{key}[{index}]", boolean


def _score(value: float) -> str:
    return f"{value:+.4f}"


def _change_lines(entry: dict[str, Any]) -> list[str]:
    task = entry["task"]
    where = f"group {entry['group_index']}" if task != boolean_sidecar.ANCHOR_TASK else "in front"
    head = f"  kept {task}, {where}"
    if "concept" in entry:
        head += f", {entry['concept']} + {json.dumps(entry['added'], ensure_ascii=False)}"
    head += f": {_score(entry['score_before'])} -> {_score(entry['score_after'])}"
    return [head, f"    before  {entry['before']}", f"    after   {entry['after']}"]


def replay(
    entries: Sequence[tuple[str, str]],
    lexicon: Lexicon,
    ask: boolean_sidecar.Ask,
    *,
    emit: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Offer each string to ``ask`` through the production per-string logic
    and print what happened. Returns the totals."""
    totals = {
        "strings": 0,
        "eligible": 0,
        "changed": 0,
        "requests": 0,
        "seconds": 0.0,
        "stopped": False,
    }
    cap = boolean_sidecar.MAX_CONCEPT_RUNS_PER_STRING
    for label, boolean in entries:
        totals["strings"] += 1
        if not parse(boolean).legal:
            emit(f"{label}  does not parse, so it is never sent")
            emit(f"  string  {boolean}")
            continue
        started = clock()
        try:
            outcome = boolean_sidecar._improve(boolean, lexicon, ask)
        except boolean_sidecar._SidecarDown:
            totals["seconds"] += clock() - started
            totals["stopped"] = True
            emit(f"{label}  the sidecar stopped answering; this string and the rest are not replayed")
            break
        except Exception as error:  # noqa: BLE001 — the hook swallows it too; say so here
            totals["seconds"] += clock() - started
            emit(f"{label}  failed inside the sidecar logic: {type(error).__name__}: {error}")
            continue
        seconds = clock() - started
        totals["seconds"] += seconds
        totals["requests"] += outcome.requests
        if outcome.whole_groups or outcome.concept_runs:
            totals["eligible"] += 1
        if outcome.kept:
            totals["changed"] += 1
        sent = min(outcome.concept_runs, cap)
        emit(
            f"{label}  whole groups {outcome.whole_groups}, concept runs "
            f"{outcome.concept_runs} ({sent} sent), requests {outcome.requests}, "
            f"{seconds:.1f} s"
        )
        emit(f"  string  {boolean}")
        for entry in outcome.kept:
            for line in _change_lines(entry):
                emit(line)
    emit("")
    emit(
        f"strings {totals['strings']}, eligible {totals['eligible']}, "
        f"changed {totals['changed']}, requests {totals['requests']}, "
        f"{totals['seconds']:.1f} s"
    )
    return totals


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state_dir", help=f"a run's state directory, holding {PLAN_FILE}")
    parser.add_argument("--url", default=None, help="sidecar address (default: the configured one)")
    parser.add_argument("--limit", type=int, default=None, help="replay only the first N strings")
    args = parser.parse_args(argv)

    plan_path = Path(args.state_dir) / PLAN_FILE
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"cannot read {plan_path}: {error}", file=sys.stderr)
        return 2
    if not isinstance(plan, dict):
        print(f"{plan_path} is not a plan object", file=sys.stderr)
        return 2
    lexicon = load_role_lexicon()
    if lexicon is None:
        print("the role lexicon will not load", file=sys.stderr)
        return 2

    entries = list(plan_strings(plan))
    if args.limit is not None:
        entries = entries[: max(args.limit, 0)]
    ask = boolean_sidecar._asker(boolean_sidecar._Breaker(), url=args.url)
    url = args.url or str(
        getattr(_config, "LINKEDIN_ENRICHER_SIDECAR_URL", "") or boolean_sidecar.DEFAULT_URL
    )
    print(f"plan {plan_path}, {len(entries)} strings, sidecar {url}, lexicon {lexicon.role}")
    print("")
    totals = replay(entries, lexicon, ask)
    return 1 if totals["stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
