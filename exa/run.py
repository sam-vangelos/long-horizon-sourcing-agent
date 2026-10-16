#!/usr/bin/env python3
"""Exa sourcing pipeline — CLI entry point.

Usage::

    python -m exa.run --brief config/<campaign>/brief.json --check
    python -m exa.run --brief config/<campaign>/brief.json
    python -m exa.run --brief config/<campaign>/brief.json --resume
    python -m exa.run --brief config/<campaign>/brief.json --resume --rejudge-stale
    python -m exa.run --brief config/<campaign>/brief.json --max-queries 5
    python -m exa.run --brief config/<campaign>/brief.json \\
        --known-from output/state/linkedin/<project-dir>

``--check`` is the preflight: it answers every question a launch can be
wrong about — does the brief lint clean, what does it compile to, is the
Exa key live, where does state land, is this a fresh run or a resume,
and what is the spend cap — and it answers them WITHOUT opening the
store, creating a directory, or spending a cent. That constraint is the
whole point: a preflight that writes state is a preflight you cannot run
twice and get the same answer from.

Exit codes: 0 success (including a deliberate pause), 1 the run errored,
2 the launch was REFUSED before it started — a bad brief, a blocking
lint finding, or a fresh/resume mismatch. The split matters because 2
means nothing happened and the operator can fix the command, while 1
means state moved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

# Import for the side effect: `.env` resolution (project root, or the
# user-data dir on a frozen build) has to happen before anything reads
# EXA_API_KEY. The module deliberately creates no directories at import
# time, so this is safe on the ``--check`` path.
import exa.config as exa_config  # noqa: F401
from exa.lane_compiler import blocking_findings, compile_exa_queries, lint_exa_brief
from shared.brief_loader import load_brief
from shared.output_paths import resolve_exa_state_dir


RUNTIME_DB_FILENAME = "runtime_state.sqlite3"
EXA_USAGE_FILENAME = "exa-usage.jsonl"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exa Sourcing Pipeline")
    parser.add_argument("--brief", required=True, help="Path to the sourcing brief JSON")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Brief-scoped state directory (default: output/state/exa/<brief-id>/)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume the existing run")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Preflight only: lint, compile, probe, report — writes nothing",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Stop after this many queries (smoke runs)",
    )
    parser.add_argument(
        "--rejudge-stale",
        action="store_true",
        help="On resume, judge again (two samples each) the short-list and "
        "worth-a-look verdicts made under earlier judging criteria",
    )
    parser.add_argument(
        "--rejudge-stale-rejects",
        action="store_true",
        help="Also judge again (one sample each) the rejects made under earlier "
        "judging criteria; implies --rejudge-stale",
    )
    parser.add_argument(
        "--known-from",
        action="append",
        default=[],
        metavar="DIR",
        help="Another campaign's state dir; its saves populate the "
        "'already known' section (report-only, never a filter)",
    )
    return parser


def _print(line: str) -> None:
    """One prebuilt console line, indent preserved (``exa/extraction.py:_one_line``)."""

    raw = str(line)
    indent = raw[: len(raw) - len(raw.lstrip(" "))]
    print(indent + " ".join(raw.split()), flush=True)


def check(args: argparse.Namespace) -> int:
    """The preflight. Returns the process exit code; writes no state."""

    brief = load_brief(args.brief)
    findings = lint_exa_brief(brief)
    for finding in findings:
        _print(f"    [exa] {finding.severity.upper()} {finding.code}: {finding.message}")
    if blocking_findings(findings):
        # Stop before the readiness probe. The probe is a real, billed Exa search
        # (``exa/health.py:probe_exa_readiness``), and a brief that cannot launch
        # has not earned one.
        _print("    [exa] NO-GO — blocking lint finding(s); nothing else was checked")
        return EXIT_REFUSED

    queries = compile_exa_queries(brief)
    firms = {query.firm for query in queries if query.firm}
    archetypes = [query for query in queries if query.lane == "archetype"]
    # Firm lanes only. The archetype lane runs no firm × lane product —
    # counting it in the "× N lanes" factor would make the arithmetic
    # of the line stop working (20 × 4 ≠ 66) and read as a bug.
    lanes = {query.lane for query in queries if query.lane != "archetype"}
    plan = (
        f"    [exa] {len(queries)} queries across {len(firms)} "
        f"{'firm' if len(firms) == 1 else 'firms'} × {len(lanes)} "
        f"{'lane' if len(lanes) == 1 else 'lanes'}"
    )
    if archetypes:
        plan += (
            f" + {len(archetypes)} "
            f"{'archetype' if len(archetypes) == 1 else 'archetypes'}"
        )
    _print(plan)

    # Lint already refused a malformed block, so this reads a valid one.
    from exa.scope_gate import CurrentEmployerScope

    scope = CurrentEmployerScope.from_brief(brief)
    if scope is not None:
        parents = {parent for _name, parent in scope.parents}
        _print(
            f"    [exa] current-employer scope check on: {len(scope.parents)} names "
            f"across {len(parents)} firms; an out-of-scope save goes to review"
        )

    from exa.location_gate import LocationScope

    location_scope = LocationScope.from_brief(brief)
    if location_scope is not None:
        _print(
            f"    [exa] location scope on: {location_scope.label}; a card location "
            "outside it is excluded before judgment, an unreadable one goes to "
            "the judge"
        )

    from exa.health import probe_exa_readiness

    report = probe_exa_readiness()
    if report.ready:
        _print("    [exa] Exa ready")
    else:
        for blocker in report.blockers:
            _print(f"    [exa] NOT READY {blocker.kind}: {blocker.message}")

    state_dir = _preview_state_dir(args)
    # Same test the launch path applies, so --check cannot report a
    # posture the launch then disagrees with: a RUN ROW, not a file.
    if _state_dir_holds_a_run(state_dir):
        posture = "holds a run — launch with --resume to continue it"
    elif (state_dir / RUNTIME_DB_FILENAME).exists():
        posture = "database but no run — this would be a fresh run"
    else:
        posture = "empty — this would be a fresh run"
    _print(f"    [exa] state dir {state_dir} ({posture})")
    for line in _rejudge_lines(args, brief, state_dir):
        _print(line)

    from shared.config import EXA_RUN_BUDGET_USD

    cap = float(EXA_RUN_BUDGET_USD)
    if cap:
        _print(f"    [exa] run budget cap ${cap:.2f}")
    else:
        _print("    [exa] run budget cap disabled (no cap)")
    _print(_spend_to_date_line(state_dir, cap))

    return EXIT_OK if report.ready else EXIT_ERROR


def _spend_to_date_line(state_dir: Path, cap: float) -> str:
    """What this campaign has spent on Exa, and where a resumed meter starts.

    Read from the receipt ledger, never from a previous session's meter:
    the meter dies with its session and can hold reservations Exa never
    billed, while the ledger holds one receipt per response. The cap is
    per session, so a resumed session's meter starts at zero whatever the
    campaign has spent. Read-only, like the rest of the preflight.
    """

    from exa.cost_meter import ledger_spend_usd

    ledger = state_dir / EXA_USAGE_FILENAME
    cap_clause = f"of the ${cap:.2f} cap" if cap else "with no cap"
    if not ledger.exists():
        return (
            f"    [exa] Exa spend to date $0.0000 (no {EXA_USAGE_FILENAME} yet); "
            f"a session's meter starts at $0.0000 {cap_clause}"
        )
    total, receipts = ledger_spend_usd(ledger)
    return (
        f"    [exa] Exa spend to date ${total or 0.0:.4f} across {receipts} "
        f"{'receipt' if receipts == 1 else 'receipts'} ({EXA_USAGE_FILENAME}); "
        f"a resumed session's meter starts at $0.0000 {cap_clause}"
    )


def _rejudge_lines(args: argparse.Namespace, brief, state_dir: Path) -> list[str]:
    """What a resume will judge again before it searches, and on what criteria.

    Two lines for a state dir that holds a run: the current brief's judgment
    fingerprint (``exa.judge.judgment_fingerprint``, the value stamped on every
    verdict this brief makes) with the re-judge opt-ins this command carries,
    then the counts (``exa/rejudge.py``) and what the opt-ins do with them,
    with the judge calls that adds up to. The second line is omitted when
    nothing is stale. Read-only, like the rest of the preflight; nothing for
    a state dir with no run.
    """

    if not _state_dir_holds_a_run(state_dir):
        return []
    from exa.judge import judgment_fingerprint
    from exa.rejudge import REJECT_SAMPLES, SLATE_SAMPLES, count_rejudge_work
    from shared.brief_identity import compute_brief_identity
    from shared.output_paths import derive_brief_id

    opt_ins = _rejudge_opt_ins(args)
    slate_on, rejects_on = opt_ins["rejudge_stale"], opt_ins["rejudge_stale_rejects"]
    fingerprint = judgment_fingerprint(brief)
    lines = [
        f"    [exa] judgment fingerprint "
        f"{fingerprint or '(none; the brief file hash stands in)'} (the current "
        "brief); re-judge on resume: slate verdicts "
        f"{'on (--rejudge-stale)' if slate_on else 'off'}, earlier rejects "
        f"{'on (--rejudge-stale-rejects)' if rejects_on else 'off'}"
    ]

    brief_id = str(
        derive_brief_id(brief_path=Path(args.brief), brief=brief)
        or Path(args.brief).stem
    )
    identity = compute_brief_identity(args.brief)
    current_hash = str(identity["brief_content_hash"]) if identity else ""
    try:
        conn = _read_only_connect(state_dir / RUNTIME_DB_FILENAME)
        try:
            conn.row_factory = sqlite3.Row
            rows = [
                {
                    "current_lifecycle_state": row["current_lifecycle_state"],
                    "terminal_decision": row["terminal_decision"],
                    "user_status": row["user_status"],
                    "brief_content_hash": row["brief_content_hash"],
                    "terminal_payload": _json_dict(row["terminal_payload_json"]),
                }
                for row in conn.execute(
                    "SELECT current_lifecycle_state, terminal_decision, user_status, "
                    "brief_content_hash, terminal_payload_json FROM candidates "
                    "WHERE source = 'exa' AND brief_id = ?",
                    (brief_id,),
                )
            ]
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return lines
    work = count_rejudge_work(
        rows, current_fingerprint=fingerprint, current_brief_hash=current_hash
    )

    def verdicts(count: int, what: str) -> str:
        return f"{count} {what} {'verdict was' if count == 1 else 'verdicts were'}"

    clauses: list[str] = []
    calls = 0
    if work.stale:
        it = "it" if work.stale == 1 else "them"
        head = f"{verdicts(work.stale, 'slate')} judged under earlier judging criteria"
        if slate_on:
            clauses.append(
                f"{head}; a resume re-judges {it} first, two samples each "
                "(model spend only, no Exa call)"
            )
            calls += SLATE_SAMPLES * work.stale
        else:
            keeps = "it keeps its" if work.stale == 1 else "they keep their"
            clauses.append(
                f"{head}; {keeps} verdict on resume (re-judge is off; "
                "--rejudge-stale turns it on)"
            )
    if work.stale_rejects:
        one = work.stale_rejects == 1
        if rejects_on:
            clauses.append(
                f"{work.stale_rejects} earlier "
                + ("reject is re-judged too, one sample" if one
                   else "rejects are re-judged too, one sample each")
            )
            calls += REJECT_SAMPLES * work.stale_rejects
        else:
            clauses.append(
                f"{work.stale_rejects} earlier "
                + ("reject keeps its verdict" if one else "rejects keep their verdict")
            )
    if work.cleared:
        clauses.append(
            f"{work.cleared} "
            + ("verdict a stopped re-judge cleared is" if work.cleared == 1
               else "verdicts a stopped re-judge cleared are")
            + " judged again, two samples each"
        )
        calls += SLATE_SAMPLES * work.cleared
    if clauses:
        lines.append(
            "    [exa] "
            + "; ".join(clauses)
            # "Up to": a person a gate now excludes is settled with no call.
            + (f" (up to {calls} judge calls)" if calls else "")
        )
    return lines


def _rejudge_opt_ins(args: argparse.Namespace) -> dict[str, bool]:
    """The re-judge pass's opt-ins; the rejects flag implies the slate one."""

    rejects = bool(getattr(args, "rejudge_stale_rejects", False))
    return {
        "rejudge_stale": rejects or bool(getattr(args, "rejudge_stale", False)),
        "rejudge_stale_rejects": rejects,
    }


def _json_dict(raw) -> dict:
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _preview_state_dir(args: argparse.Namespace) -> Path:
    """Where state WOULD land, without creating it.

    ``resolve_exa_state_dir`` mkdirs, which a preflight must not do — a
    ``--check`` that leaves an empty directory behind makes the next
    ``--check`` report a different posture than the first.
    """

    if args.state_dir:
        return Path(args.state_dir).resolve()
    from shared.config import OUTPUT_DIR
    from shared.output_paths import exa_state_key, slugify_output_component

    key = exa_state_key(brief_path=args.brief)
    return Path(OUTPUT_DIR) / "state" / slugify_output_component("exa") / key


def _state_dir_holds_a_run(state_dir: Path) -> bool:
    """Does this state dir hold an actual RUN, not just a database file?

    The distinction is the whole point. A database file proves the store
    was opened; a row in ``runs`` proves a session started. A crash
    between ``initialize()`` and the first ``start_run`` leaves the
    former without the latter, and the two must not be confused: file
    presence would tell a resume there is something to continue when
    there is nothing, and tell a fresh launch it is about to clobber
    work that was never done.

    Read-only — a posture check must never migrate or write the store it
    is inspecting — and missing, corrupt, and empty all collapse to "no
    run", the semantics ``shared.runtime_state.read_models
    .latest_run_summary`` gives. A corrupt DB therefore reads as "no run",
    which is the safe direction: the run's own store open will raise on it
    with a real error instead of this check guessing at one.
    """

    db_path = state_dir / RUNTIME_DB_FILENAME
    if not db_path.exists():
        return False

    try:
        conn = _read_only_connect(db_path)
        try:
            row = conn.execute(
                "SELECT id FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
    return row is not None and row[0] is not None


def _read_only_connect(db_path: Path) -> sqlite3.Connection:
    """Open the store for reading without writing a byte into the state dir.

    ``mode=ro`` alone still WRITES: opening a WAL-mode database creates its
    ``-wal`` and ``-shm`` siblings when they are absent, which put two new
    files into the state dir of every closed campaign a preflight looked
    at. With no ``-wal`` beside it the main file is the whole database, so
    ``immutable=1`` reads it without creating anything; with one, a writer
    is live or crashed mid-session, both siblings already exist, and the
    ordinary read-only open sees the uncheckpointed rows.
    """

    wal = db_path.with_name(db_path.name + "-wal")
    uri = f"file:{db_path}?mode=ro" + ("" if wal.exists() else "&immutable=1")
    return sqlite3.connect(uri, uri=True)


def _launch_posture(args: argparse.Namespace) -> tuple[str | None, bool]:
    """``(refusal, effective_resume)`` for this fresh-vs-resume command.

    Fresh-vs-resume is a CAMPAIGN decision and the operator's to make: a fresh run
    over a live state dir re-runs a query plan somebody already paid for,
    and a resume against an empty dir is a typo that would silently
    become a fresh launch. Both are refused with the command that fixes
    them, before any state is touched.

    The third case is the one that is NOT a refusal. A state dir holding
    a database with no run rows is what a crash between store
    initialization and the first ``start_run`` leaves behind. There is
    nothing there to resume — no query plan, no candidates, no verdicts
    — so refusing ``--resume`` would strand the campaign on an operator
    who has to know to delete a file, and honoring it would run
    "resume" semantics over a plan that was never written. It runs
    FRESH, and says so in one line, because a mode the operator did not
    ask for is not allowed to be silent.
    """

    state_dir = _preview_state_dir(args)
    db_exists = (state_dir / RUNTIME_DB_FILENAME).exists()
    holds_run = _state_dir_holds_a_run(state_dir)

    if args.resume and not holds_run:
        if db_exists:
            _print(
                "    [exa] state dir has a database but no run — "
                "starting fresh"
            )
            return (None, False)
        return (
            f"NO-GO — resume requested but {state_dir} has no "
            f"{RUNTIME_DB_FILENAME}",
            False,
        )
    if not args.resume and holds_run:
        return (f"NO-GO — {state_dir} already holds a run; use --resume", False)
    return (None, bool(args.resume))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.check:
            return check(args)
    except Exception as exc:  # noqa: BLE001 — a bad brief is a refusal, not a crash
        _print(f"    [exa] NO-GO — cannot read the brief: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    refusal, effective_resume = _launch_posture(args)
    if refusal:
        _print(f"    [exa] {refusal}")
        return EXIT_REFUSED

    from exa.orchestrator import (
        ExaBriefBlockedError,
        ExaPipeline,
        ExaStateDirBusyError,
    )

    try:
        state_dir = resolve_exa_state_dir(
            brief_path=args.brief, state_dir=args.state_dir
        )
        pipeline = ExaPipeline(
            brief_path=args.brief,
            output_dir=state_dir,
            known_from=[Path(p) for p in (args.known_from or [])],
            **_rejudge_opt_ins(args),
        )
    except Exception as exc:  # noqa: BLE001 — brief load / parse failures
        _print(f"    [exa] NO-GO — cannot read the brief: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    try:
        stats = asyncio.run(
            pipeline.run(resume=effective_resume, max_queries=args.max_queries)
        )
    except ExaBriefBlockedError:
        # The blocking findings already printed one line each.
        return EXIT_REFUSED
    except ExaStateDirBusyError:
        # The NO-GO line already printed.
        return EXIT_REFUSED

    summary = (
        f"    [exa] run {stats.run_id} — {stats.judged} judged, "
        f"{stats.saved} saved, {stats.review} worth a look, "
        f"{stats.rejected} rejected"
    )
    gated = int(getattr(stats, "blacklist_gated", 0) or 0)
    if gated:
        summary += (
            f", {gated} excluded before judgment "
            "(current employer on the blacklist)"
        )
    alumni_gated = int(getattr(stats, "alumni_gated", 0) or 0)
    if alumni_gated:
        summary += (
            f", {alumni_gated} excluded before judgment "
            "(worked at an alumni-blacklisted employer)"
        )
    location_gated = int(getattr(stats, "location_gated", 0) or 0)
    if location_gated:
        scope = getattr(pipeline, "location_scope", None)
        label = scope.label if scope is not None else "the brief's location scope"
        summary += f", {location_gated} excluded before judgment (outside {label})"
    summary += _rejudge_clause(stats, pipeline)
    _print(summary)
    return EXIT_OK


def _rejudge_clause(stats, pipeline) -> str:
    """The run summary's re-judge clause, or "" when nothing was re-judged.

    "Left the short list" counts the judge's (and the scope check's) moves
    only. People a deterministic gate removed on re-judge are reported on
    their own, so the count reads as what the new criteria did.
    """

    def count(name: str) -> int:
        return int(getattr(stats, name, 0) or 0)

    rejudged = count("rejudged")
    kept = count("rejudge_kept")
    if not (rejudged or kept):
        return ""
    clause = ""
    if rejudged:
        detail = [f"{count('rejudge_left_short_list')} left the short list"]
        if count("rejudge_split"):
            detail.append(f"{count('rejudge_split')} split to worth a look")
        if count("rejudge_rejects"):
            detail.append(
                f"{count('rejudge_promoted')} of {count('rejudge_rejects')} earlier "
                "rejects promoted"
            )
        clause += (
            f", {rejudged} re-judged under the current criteria ({', '.join(detail)})"
        )
        location = count("rejudge_location_excluded")
        if location:
            scope = getattr(pipeline, "location_scope", None)
            label = scope.label if scope is not None else "the brief's location scope"
            clause += f", {location} of them excluded on re-judge (outside {label})"
        blacklist = count("rejudge_blacklist_excluded")
        if blacklist:
            clause += f", {blacklist} of them excluded on re-judge (blacklisted employer)"
    if kept:
        clause += (
            f", {kept} kept {'its' if kept == 1 else 'their'} stored verdict after a "
            "failed re-judge sample (tried again next resume)"
        )
    return clause


if __name__ == "__main__":
    sys.exit(main())
