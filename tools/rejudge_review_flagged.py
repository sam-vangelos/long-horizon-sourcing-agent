"""Re-judge a finished campaign's stored profiles against the current seed brief.

Offline re-adjudication lane: after a brief revision (a recalibrated seed, a
hiring-manager intake), re-decide candidates the campaign already read through
the REAL judgment path — preflight_v2 generates the structured criteria from
the current seed, then full_judge runs on each candidate's stored profile
summary. No browser, no state-dir writes: results land in an archive rescue
dir for the save pass to consume.

Selection is the latest verdict per profile_url in final_judgments.jsonl,
filtered by --decisions (default REVIEW_FLAGGED, the original rescue lane;
`all` takes every latest verdict). Each result row records the candidate's
real prior decision in old_decision beside the new one. REVIEW_FLAGGED is
retired from the judgment contract, so a new verdict is always a save-family
decision or REJECT — never another deferral.

--dry-run prints the selection per old decision and how many have a stored
profile, then exits before any model call: it prices a run before spending.

--resume <rescue-dir> finishes a killed run on the criteria it already
generated: the dir's preflight_v2_brief.json loads through the same V2 loader
finalize_preflight_v2 used (no new preflight call — a fresh one would produce
different criteria), every URL with a real new_decision in the dir is skipped
(RESCUE_ERROR rows are re-attempted), and new rows append in that dir.
--shard i/n (needs --resume, so every shard judges on one criteria set) takes
the remaining URLs whose stable hash lands in slot i, writing to
rejudgments.shard-i-of-n.jsonl; n concurrent processes cover each URL once.
--merge <rescue-dir> folds rejudgments.jsonl and the shard files into
rejudgments.merged.jsonl and rebuilds passers.json, with no model call.

The brief's deterministic exclusions run before any judge call, as they do in
the live run: someone on the prior_search skip list, current staff of an
employer_blacklist name, and anyone who has EVER worked at an alumni_blacklist
name are recorded as GATED and never reach passers.json. A passer whose search
card shows a location in the brief's excluded_locations goes to
passers-excluded-location.json instead (another recruiter's market), never to
passers.json, which the save pass consumes.

Usage:
    .venv/bin/python tools/rejudge_review_flagged.py <state-dir> \
        [--decisions REVIEW_FLAGGED|REJECT,SAVE|all] [--limit N] [--dry-run] \
        [--resume <rescue-dir> [--shard i/n]]
    .venv/bin/python tools/rejudge_review_flagged.py --merge <rescue-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from linkedin.orchestrator import load_brief  # noqa: E402
from shared import config  # noqa: E402
from shared.judger import full_judge, init_judger  # noqa: E402
from shared.preflight_v2 import (  # noqa: E402
    finalize_preflight_v2,
    generate_preflight_v2_once,
)
from shared.schemas import CandidateProfileSummary  # noqa: E402
from shared.employer_blacklist import (  # noqa: E402
    match_any_employer,
    match_current_employer,
)
from shared.excluded_locations import excluded_location_match  # noqa: E402

# Imported, not re-declared: the save family is the contract's, and a new
# save channel must reach the rescue pass automatically.
from shared.contracts import SAVE_DECISIONS  # noqa: E402


def parse_decisions(raw: str) -> set[str] | None:
    """Upper-cased decision names from a comma list; None means `all`."""
    names = {part.strip().upper() for part in raw.split(",") if part.strip()}
    return None if "ALL" in names else names


def latest_by_decision(
    final_judgments_path: Path, decisions: set[str] | None
) -> dict[str, dict]:
    """Latest verdict per profile_url; keep those whose latest decision is selected.

    The filter runs on the LATEST row only: a URL rejected once and saved on a
    later pass counts as SAVE. Stored decisions are the contract's upper-case
    names, so they compare directly. `decisions=None` keeps every latest verdict.
    """
    latest: dict[str, dict] = {}
    for line in final_judgments_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        url = row.get("profile_url")
        if url:
            latest[url] = row
    if decisions is None:
        return latest
    return {u: r for u, r in latest.items() if r.get("decision") in decisions}


def load_summaries(profile_summaries_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in profile_summaries_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        url = row.get("profile_url")
        if url:
            out[url] = row  # later rows win — latest extraction
    return out


def load_snippets(snippets_path: Path) -> dict[str, dict]:
    """The latest search card per profile_url; empty when the file is absent."""
    out: dict[str, dict] = {}
    if not snippets_path.exists():
        return out
    for line in snippets_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        url = row.get("profile_url") if isinstance(row, dict) else None
        if url:
            out[url] = row
    return out


def deterministic_gate(url: str, summary: CandidateProfileSummary, card: dict,
                       brief) -> str | None:
    """Why the brief's exclusions settle this person with no judge call, else None.

    The same three exclusions the live run applies before judgment: the skip
    list of people already contacted, current staff of a blacklisted employer,
    and anyone who has ever worked at an alumni-blacklisted one.
    """
    prior_search = getattr(brief, "prior_search", None)
    ruled_out = set(getattr(prior_search, "ruled_out_urls", None) or [])
    if url in ruled_out:
        return "prior_search: already contacted outside this run"
    experiences = summary.experiences or []
    current_company = str(card.get("current_company") or "") or (
        experiences[0].company if experiences else ""
    )
    hit = match_current_employer(
        current_company, summary.headline, getattr(brief, "employer_blacklist", None)
    )
    if hit:
        return f"employer_blacklist: {hit[0]} ({hit[1]})"
    hit = match_any_employer(
        getattr(brief, "alumni_blacklist", None),
        current_company=current_company,
        headline=summary.headline,
        experience_companies=[e.company for e in experiences],
        experience_entries=[
            *(card.get("experience_entries") or ()),
            *(e.title for e in experiences),
        ],
        about=summary.about or "",
    )
    if hit:
        return f"alumni_blacklist: {hit[0]} ({hit[1]})"
    return None


def excluded_location_of(card: dict, brief) -> str:
    """The brief's excluded_locations entry the card's location falls in, or ""."""
    raw = getattr(brief, "raw", None)
    excluded = raw.get("excluded_locations") if isinstance(raw, dict) else None
    return excluded_location_match(
        str(card.get("location") or ""), excluded if isinstance(excluded, list) else None
    ) or ""


EXCLUDED_LOCATION_SUFFIX = "-excluded-location"


def write_passers(passers_path: Path, passers: list[dict]) -> tuple[int, int]:
    """passers.json gets in-territory passers only; the rest go beside it.

    Returns (in territory, excluded location). The save pass reads
    passers.json, so a passer in another recruiter's market is never saved
    by accident; the sibling file is the hand-off.
    """
    kept = [r for r in passers if not r.get("excluded_location")]
    elsewhere = [r for r in passers if r.get("excluded_location")]
    passers_path.write_text(json.dumps(kept, indent=2))
    sibling = passers_path.with_name(
        passers_path.stem + EXCLUDED_LOCATION_SUFFIX + passers_path.suffix
    )
    if elsewhere or sibling.exists():
        sibling.write_text(json.dumps(elsewhere, indent=2))
    return len(kept), len(elsewhere)


GATED = "GATED"
RESCUE_ERROR = "RESCUE_ERROR"
PREFLIGHT_FILE = "preflight_v2_brief.json"
RESULTS_FILE = "rejudgments.jsonl"
MERGED_FILE = "rejudgments.merged.jsonl"
_SHARD_FILE = re.compile(r"^rejudgments\.shard-(\d+)-of-(\d+)\.jsonl$")


def parse_shard(raw: str) -> tuple[int, int]:
    """`i/n` with 1 <= i <= n; argparse turns the ValueError into a usage error."""
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", raw)
    if not match:
        raise argparse.ArgumentTypeError(f"--shard wants i/n (e.g. 1/4), got {raw!r}")
    index, total = int(match.group(1)), int(match.group(2))
    if total < 1 or not 1 <= index <= total:
        raise argparse.ArgumentTypeError(f"--shard {raw!r}: need 1 <= i <= n")
    return index, total


def shard_of(profile_url: str, total: int) -> int:
    """1-based shard slot for a URL: a stable hash, never list position or
    Python's per-process salted hash(), so every process agrees."""
    digest = hashlib.sha256(profile_url.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % total + 1


def shard_files(rescue_dir: Path) -> list[Path]:
    """Shard result files in the dir, ordered by (n, i)."""
    found = []
    for path in rescue_dir.iterdir():
        match = _SHARD_FILE.match(path.name)
        if match:
            found.append(((int(match.group(2)), int(match.group(1))), path))
    return [path for _, path in sorted(found)]


def read_result_rows(path: Path) -> tuple[list[dict], int]:
    """Rows of a results file plus a count of unreadable lines: a kill can tear
    the last row mid-write, and that URL must be re-attempted, not crash us."""
    rows: list[dict] = []
    torn = 0
    if not path.exists():
        return rows, torn
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
            continue
        if isinstance(row, dict) and row.get("profile_url"):
            rows.append(row)
        else:
            torn += 1
    return rows, torn


def is_real_decision(row: dict) -> bool:
    decision = row.get("new_decision")
    return bool(decision) and decision != RESCUE_ERROR


def merge_rows(rows: list[dict]) -> dict[str, dict]:
    """One row per URL, in first-seen order: the last real decision wins; an
    error row stands only for a URL no real decision ever reached."""
    merged: dict[str, dict] = {}
    for row in rows:
        url = row["profile_url"]
        prior = merged.get(url)
        if prior is None or is_real_decision(row) or not is_real_decision(prior):
            merged[url] = row
    return merged


def rescue_rows(rescue_dir: Path) -> tuple[list[dict], int]:
    """Every row in the dir: the unsharded file first, then each shard file."""
    rows, torn = read_result_rows(rescue_dir / RESULTS_FILE)
    for path in shard_files(rescue_dir):
        more, more_torn = read_result_rows(path)
        rows += more
        torn += more_torn
    return rows, torn


def passers_from(rows: list[dict]) -> list[dict]:
    return [row for row in merge_rows(rows).values()
            if row.get("new_decision") in SAVE_DECISIONS]


def run_merge(rescue_dir: Path) -> int:
    """--merge: fold every results file into one, rebuild passers.json."""
    rows, torn = rescue_rows(rescue_dir)
    merged = merge_rows(rows)
    with (rescue_dir / MERGED_FILE).open("w") as fh:
        for row in merged.values():
            fh.write(json.dumps(row) + "\n")
    passers = [r for r in merged.values() if r.get("new_decision") in SAVE_DECISIONS]
    kept, elsewhere = write_passers(rescue_dir / "passers.json", passers)
    errors = sum(not is_real_decision(r) for r in merged.values())
    sources = [RESULTS_FILE] + [p.name for p in shard_files(rescue_dir)]
    print(f"[merge] {len(merged)} candidates from {len(rows)} rows in "
          f"{', '.join(sources)}; passers: {kept} in territory, {elsewhere} in "
          f"an excluded location; still errored: {errors}"
          + (f"; unreadable lines skipped: {torn}" if torn else ""))
    print(f"[merge] merged results: {rescue_dir / MERGED_FILE}")
    print(f"[merge] passers: {rescue_dir / 'passers.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_dir", type=Path, nargs="?")
    ap.add_argument("--brief", type=Path, default=None,
                    help="Seed brief path (default: the Brazil seed)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Judge at most N candidates (0 = all)")
    ap.add_argument("--decisions", default="REVIEW_FLAGGED",
                    help="Comma-separated prior decisions to re-judge, matched "
                         "case-insensitively against each URL's LATEST verdict "
                         "(e.g. REJECT,SAVE); `all` selects every candidate. "
                         "Default: REVIEW_FLAGGED.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the selection per old decision and how many "
                         "have a stored profile, then exit: no preflight, no "
                         "judge call, nothing written.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Archive dir for the rescue artifacts (default: the "
                         "Brazil project archive, where this lane was born). "
                         "Point it at the campaign you are rescuing — nothing "
                         "here writes to a live state dir either way.")
    ap.add_argument("--resume", type=Path, default=None, metavar="RESCUE_DIR",
                    help="Finish a killed run: judge on RESCUE_DIR's saved "
                         "preflight_v2_brief.json (no new preflight call), skip "
                         "every URL it already holds a real decision for "
                         "(RESCUE_ERROR rows are re-attempted), write there.")
    ap.add_argument("--shard", type=parse_shard, default=None, metavar="I/N",
                    help="With --resume: take only the remaining URLs whose "
                         "stable hash lands in shard I of N, writing "
                         "rejudgments.shard-I-of-N.jsonl. Run N processes, "
                         "then --merge.")
    ap.add_argument("--merge", type=Path, default=None, metavar="RESCUE_DIR",
                    help="No model calls: fold RESCUE_DIR's rejudgments.jsonl and "
                         "shard files into rejudgments.merged.jsonl (last real "
                         "decision per URL wins) and rebuild passers.json.")
    args = ap.parse_args()

    if args.merge is not None:
        if args.state_dir is not None or args.resume or args.shard:
            ap.error("--merge takes only a rescue dir")
        if not args.merge.is_dir():
            ap.error(f"--merge: no such dir {args.merge}")
        return run_merge(args.merge)
    if args.state_dir is None:
        ap.error("state_dir is required (unless --merge)")
    if args.shard and args.resume is None:
        ap.error("--shard needs --resume <rescue-dir>: every shard must judge on "
                 "the same saved criteria, and a fresh preflight differs per call")
    if args.resume is not None:
        if args.brief is not None or args.out_dir is not None:
            ap.error("--resume judges on the rescue dir's saved criteria and writes "
                     "there; --brief and --out-dir do not apply")
        if not (args.resume / PREFLIGHT_FILE).is_file():
            ap.error(f"--resume: {args.resume / PREFLIGHT_FILE} not found")

    brief_path = args.brief
    decisions = parse_decisions(args.decisions)
    label = "any decision" if decisions is None else " or ".join(sorted(decisions))
    selected = latest_by_decision(args.state_dir / "final_judgments.jsonl", decisions)
    summaries = load_summaries(args.state_dir / "profile_summaries.jsonl")
    items = list(selected.items())
    already: set[str] = set()
    torn = 0
    if args.resume is not None:
        prior_rows, torn = rescue_rows(args.resume)
        already = {r["profile_url"] for r in prior_rows if is_real_decision(r)}
        items = [(url, row) for url, row in items if url not in already]
    remaining = len(items)
    if args.shard:
        index, total = args.shard
        items = [(url, row) for url, row in items if shard_of(url, total) == index]
    share = len(items)
    if args.limit:
        items = items[: args.limit]

    if args.dry_run:
        # Pricing only: read the two jsonl files, print, exit — the brief is
        # never loaded and no out dir is made, so nothing is spent or written.
        by_old: dict[str, list[str]] = {}
        for url, row in selected.items():
            by_old.setdefault(row.get("decision") or "<none>", []).append(url)
        print(f"[dry-run] {len(selected)} candidates selected "
              f"(latest verdict of {label}) in {args.state_dir}")
        for old in sorted(by_old, key=lambda d: (-len(by_old[d]), d)):
            urls = by_old[old]
            stored = sum(u in summaries for u in urls)
            print(f"  {old}: {len(urls)} selected, {stored} with a stored profile")
        for name in sorted((decisions or set()) - set(by_old)):
            print(f"  {name}: 0 selected — no latest verdict carries it")
        if args.resume is not None:
            judged_here = sum(url in already for url in selected)
            print(f"[dry-run] resume from {args.resume}: already judged: "
                  f"{judged_here}; remaining: {remaining}"
                  + (f"; {len(already) - judged_here} judged URLs outside this "
                     f"selection" if len(already) > judged_here else "")
                  + (f"; unreadable lines skipped: {torn}" if torn else ""))
        if args.shard:
            index, total = args.shard
            print(f"[dry-run] shard {index}/{total}: {share} of {remaining} remaining")
        judged = sum(url in summaries for url, _ in items)
        limit_note = f" (--limit {args.limit})" if args.limit else ""
        print(f"[dry-run] judge calls a real run would make: {judged}{limit_note}; "
              f"skipped for no stored profile: {len(items) - judged}")
        return 0

    if args.resume is not None:
        return judge_and_record(args, selected, label, summaries, items,
                                already=len(already), remaining=remaining)

    if brief_path is None:
        ap.error("--brief is required: the seed brief the rescue re-judges against")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (args.out_dir or REPO / "output/archive/linkedin/9000000010") / (
        f"review-flag-rescue-{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rescue] {len(selected)} candidates with a latest verdict of {label}")

    brief = load_brief(str(brief_path))
    print(f"[rescue] preflight_v2 generating criteria from {brief_path.name} "
          f"({config.STRATEGY_MODEL_NAME})")
    generation = generate_preflight_v2_once(
        brief,
        model_name=config.STRATEGY_MODEL_NAME,
        usage_context={"stage": "rescue_preflight", "brief_id": brief.id},
    )
    execution = finalize_preflight_v2(brief, generation)
    (out_dir / PREFLIGHT_FILE).write_text(
        json.dumps(execution.brief_json, indent=2)
    )
    brief = execution.brief
    init_judger(brief)
    print(f"[rescue] V2 brief ready: "
          f"{len(execution.brief_json.get('capability_areas', []))} capability areas; "
          f"generated brief saved to {out_dir / PREFLIGHT_FILE}")

    results_path = out_dir / RESULTS_FILE
    cards = load_snippets(args.state_dir / "snippets.jsonl")
    with results_path.open("w") as fh:
        counts, missing = judge_items(items, summaries, brief, fh, cards=cards)

    passers = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if json.loads(line)["new_decision"] in SAVE_DECISIONS
    ]
    kept, elsewhere = write_passers(out_dir / "passers.json", passers)
    print(f"\n[rescue] done: {counts} | no stored profile: {missing}")
    print(f"[rescue] passers: {kept} in territory → {out_dir / 'passers.json'}; "
          f"{elsewhere} in an excluded location (hand-off, never saved here)")
    print(f"[rescue] full results: {results_path}")
    return 0


def judge_and_record(args, selected, label, summaries, items, *,
                     already: int, remaining: int) -> int:
    """--resume [--shard]: judge the remaining URLs on the saved criteria."""
    rescue_dir: Path = args.resume
    if args.shard:
        index, total = args.shard
        results_path = rescue_dir / f"rejudgments.shard-{index}-of-{total}.jsonl"
        passers_path = rescue_dir / f"passers.shard-{index}-of-{total}.json"
        scope = f"shard {index}/{total}: {len(items)} of {remaining} remaining"
    else:
        results_path = rescue_dir / RESULTS_FILE
        passers_path = rescue_dir / "passers.json"
        scope = f"{remaining} remaining"
    print(f"[rescue] {len(selected)} candidates with a latest verdict of {label}; "
          f"resuming {rescue_dir}: {already} already judged, {scope}")

    # The saved JSON is finalize_preflight_v2's brief_json verbatim; load_brief
    # routes it through _load_v2_brief, the constructor finalize itself used,
    # so the judge sees the killed run's criteria without a new preflight call.
    saved = rescue_dir / PREFLIGHT_FILE
    brief = load_brief(str(saved))
    init_judger(brief)
    print(f"[rescue] V2 brief reloaded from {saved} (no preflight call)")

    # A kill mid-write can leave the last row without its newline; start the
    # appended rows on a fresh line so every new row stays parseable.
    if results_path.exists() and results_path.stat().st_size:
        with results_path.open("rb") as fh:
            fh.seek(-1, 2)
            needs_newline = fh.read(1) != b"\n"
    else:
        needs_newline = False
    cards = load_snippets(args.state_dir / "snippets.jsonl")
    with results_path.open("a") as fh:
        if needs_newline:
            fh.write("\n")
        counts, missing = judge_items(items, summaries, brief, fh, cards=cards)

    rows, _torn = read_result_rows(results_path)
    passers = passers_from(rows)
    kept, elsewhere = write_passers(passers_path, passers)
    print(f"\n[rescue] done: {counts} | no stored profile: {missing}")
    print(f"[rescue] passers in {results_path.name}: {kept} in territory → "
          f"{passers_path}; {elsewhere} in an excluded location")
    print(f"[rescue] full results: {results_path}")
    if shard_files(rescue_dir):
        print(f"[rescue] shard files present — fold them with: --merge {rescue_dir}")
    return 0


def judge_items(items, summaries, brief, fh, *, cards=None) -> tuple[dict[str, int], int]:
    """Judge each item on `brief`, one flushed row per candidate into `fh`.

    The brief's deterministic exclusions settle a person as GATED before any
    judge call; every row carries the excluded_locations entry its card falls
    in ("" when in territory), which decides which passer file it lands in.
    """
    counts: dict[str, int] = {}
    missing = 0
    cards = cards or {}
    for i, (url, old_row) in enumerate(items, 1):
        raw_summary = summaries.get(url)
        if raw_summary is None:
            missing += 1
            print(f"  [{i}/{len(items)}] NO STORED PROFILE — {old_row.get('candidate_name')} ({url})")
            continue
        summary = CandidateProfileSummary.from_dict(raw_summary)
        card = cards.get(url) or {}
        location_entry = excluded_location_of(card, brief)
        gate = deterministic_gate(url, summary, card, brief)
        if gate:
            counts[GATED] = counts.get(GATED, 0) + 1
            fh.write(json.dumps({
                "profile_url": url,
                "candidate_name": summary.name,
                "old_decision": old_row.get("decision"),
                "old_rationale": old_row.get("rationale", ""),
                "new_decision": GATED,
                "new_rationale": gate,
                "excluded_location": location_entry,
            }) + "\n")
            fh.flush()
            print(f"  [{i}/{len(items)}]      {GATED:18s} {summary.name} — {gate}")
            continue
        try:
            verdict = full_judge(summary, brief=brief)
            decision = verdict.decision
            record = {
                "profile_url": url,
                "candidate_name": summary.name,
                "old_decision": old_row.get("decision"),
                "old_rationale": old_row.get("rationale", ""),
                "new_decision": decision,
                "new_rationale": getattr(verdict, "rationale", ""),
                "post_save_modifier": getattr(verdict, "post_save_modifier", ""),
                "confidence": getattr(verdict, "confidence", None),
                "excluded_location": location_entry,
            }
        except Exception as exc:  # judgment path may raise on API faults
            decision = RESCUE_ERROR
            record = {
                "profile_url": url,
                "candidate_name": summary.name,
                "old_decision": old_row.get("decision"),
                "new_decision": decision,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3),
            }
        counts[decision] = counts.get(decision, 0) + 1
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        marker = "PASS" if decision in SAVE_DECISIONS else "    "
        print(f"  [{i}/{len(items)}] {marker} {decision:18s} {summary.name}")
    return counts, missing


if __name__ == "__main__":
    raise SystemExit(main())
