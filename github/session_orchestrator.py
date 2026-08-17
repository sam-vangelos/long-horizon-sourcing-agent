#!/usr/bin/env python3
"""GitHub Session Orchestrator — manages multi-session GitHub sourcing cycles.

Simplified version of session_orchestrator.py. No decoy agent needed (no browser,
no anti-detection). Manages session limits and multi-session cycling.

Usage:
    python3 github_session_orchestrator.py --brief config/brief-fdl-brazil-v3.json
    python3 github_session_orchestrator.py --brief config/brief-fdl-brazil-v3.json --single-session
    python3 github_session_orchestrator.py --brief config/brief-fdl-brazil-v3.json --resume
    python3 github_session_orchestrator.py --status
"""

import argparse
import asyncio
import math
import random
import signal
import sys
import time

from shared.output_paths import resolve_github_state_dir

from github.client import GitHubAuthError
from github.governor import (
    GitHubGovernor,
    GitHubGovernorLimitReached,
    get_sessions_today,
    record_session_start,
    record_session_end,
    print_status,
    MAX_SESSIONS_PER_DAY,
)
from github.orchestrator import GitHubPipeline
from shared.failures import ApiBudgetExhaustedError


# ---------------------------------------------------------------------------
# Session duration sampling (same log-normal pattern as LinkedIn agent)
# ---------------------------------------------------------------------------

def _sample_session_duration() -> float:
    """Log-normal session duration: median ~2h, range 1.5-3h (seconds)."""
    mu = math.log(7200)
    sigma = 0.3
    raw = math.exp(mu + sigma * random.gauss(0, 1))
    return max(5400, min(10800, raw))


def _sample_cooldown_duration() -> float:
    """Cooldown between sessions: median ~30 min, range 15-60 min (seconds)."""
    mu = math.log(1800)
    sigma = 0.3
    raw = math.exp(mu + sigma * random.gauss(0, 1))
    return max(900, min(3600, raw))


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print(msg: str):
    print(f"[github-orchestrator] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

async def _run_session(brief_path: str, output_dir: str | None, resume: bool) -> dict:
    """Run a single GitHub sourcing session."""
    pipeline = GitHubPipeline(
        brief_path=brief_path,
        output_dir=output_dir,
    )
    return await pipeline.run(resume=resume)


# ---------------------------------------------------------------------------
# Multi-session cycle
# ---------------------------------------------------------------------------

async def run_day_cycle(
    brief_path: str,
    output_dir: str | None = None,
    single_session: bool = False,
    resume: bool = False,
):
    """Run the full day cycle: session → cooldown → session → cooldown → ..."""
    _print(f"Starting day cycle. Brief: {brief_path}")

    session_count = 0
    force_shutdown = False
    exit_code = 0

    def _handler(sig, frame):
        nonlocal force_shutdown
        if force_shutdown:
            _print("Force shutdown!")
            sys.exit(1)
        force_shutdown = True
        _print("Shutdown requested — finishing after current session...")

    signal.signal(signal.SIGINT, _handler)

    while not force_shutdown:
        # Check session cap
        sessions_today = get_sessions_today()
        if sessions_today >= MAX_SESSIONS_PER_DAY:
            _print(f"Daily session cap reached ({sessions_today}/{MAX_SESSIONS_PER_DAY}). Done.")
            break

        # Governor pre-check
        governor = GitHubGovernor()
        ok, reason = governor.can_start_session()
        if not ok:
            _print(f"Cannot start session: {reason}")
            break

        # Start session
        session_num = record_session_start()
        governor.start_session()
        session_count += 1
        _print(f"Session {session_num} starting (#{session_count} this cycle)")
        end_reason = "normal"
        fatal_stop = False

        try:
            stats = await _run_session(brief_path, output_dir, resume=(resume and session_count == 1))
        except GitHubAuthError as e:
            _print(f"Fatal GitHub auth error: {e}")
            stats = {"saved": 0, "enrichments": 0, "candidates_enriched": 0}
            end_reason = "github_auth_error"
            fatal_stop = True
            exit_code = 1
        except GitHubGovernorLimitReached as e:
            _print(f"Session ended: {e.reason}")
            stats = {"saved": 0, "enrichments": 0}
            end_reason = e.reason
        except ApiBudgetExhaustedError as e:
            _print(f"API budget exhausted — stopping day cycle (no retry): {e}")
            stats = {"saved": 0, "enrichments": 0, "candidates_enriched": 0}
            end_reason = "api_budget_exhausted"
            fatal_stop = True
            exit_code = 1
        except Exception as e:
            _print(f"Session error: {e}")
            stats = {"saved": 0, "enrichments": 0}
            end_reason = "error"

        # End session — use stats from the pipeline (governor in session_orchestrator
        # doesn't track enrichments; the pipeline's internal governor does)
        record_session_end(
            session_num,
            enrichments=stats.get("candidates_enriched", 0),
            reason=end_reason,
            stats=stats,
        )
        _print(f"Session {session_num} complete: {stats.get('saved', 0)} saves")

        if fatal_stop:
            break

        if single_session:
            _print("Single session mode — done.")
            break

        if force_shutdown:
            break

        # Cooldown between sessions
        cooldown = _sample_cooldown_duration()
        _print(f"Cooling down for {cooldown/60:.0f} minutes before next session...")
        try:
            await asyncio.sleep(cooldown)
        except asyncio.CancelledError:
            break

    _print(f"Day cycle complete. {session_count} sessions run.")
    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GitHub Session Orchestrator")
    parser.add_argument("--brief", help="Path to sourcing brief JSON")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Mutable brief-scoped state directory (default: output/state/github/<brief-id>/)",
    )
    parser.add_argument("--output-dir", default=None, help="Deprecated alias for --state-dir")
    parser.add_argument("--single-session", action="store_true", help="Run one session only")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress")
    parser.add_argument("--status", action="store_true", help="Print current stats")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if not args.brief:
        parser.error("--brief is required (or use --status)")

    state_dir = None
    if args.brief:
        state_dir = str(
            resolve_github_state_dir(
                brief_path=args.brief,
                state_dir=args.state_dir or args.output_dir,
            )
        )

    exit_code = asyncio.run(run_day_cycle(
        brief_path=args.brief,
        output_dir=state_dir,
        single_session=args.single_session,
        resume=args.resume,
    ))
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
