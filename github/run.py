#!/usr/bin/env python3
"""GitHub sourcing pipeline — CLI entry point.

Standalone runner (no session orchestrator). For direct pipeline runs.

Usage:
    python3 github_run.py --brief config/brief-fdl-brazil-v3.json
    python3 github_run.py --brief config/brief-fdl-brazil-v3.json --resume
    python3 github_run.py --status
"""

import argparse
import asyncio
import sys

from shared.output_paths import resolve_github_state_dir


def main():
    parser = argparse.ArgumentParser(description="GitHub Sourcing Pipeline")
    parser.add_argument("--brief", help="Path to sourcing brief JSON")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Mutable brief-scoped state directory (default: output/state/github/<brief-id>/)",
    )
    parser.add_argument("--output-dir", default=None, help="Deprecated alias for --state-dir")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress")
    parser.add_argument("--status", action="store_true", help="Print current session stats")

    args = parser.parse_args()

    if args.status:
        from github.governor import print_status
        print_status()
        return

    if not args.brief:
        parser.error("--brief is required (or use --status)")

    from github.orchestrator import GitHubPipeline

    state_dir = resolve_github_state_dir(
        brief_path=args.brief,
        state_dir=args.state_dir or args.output_dir,
    )
    pipeline = GitHubPipeline(
        brief_path=args.brief,
        output_dir=str(state_dir),
    )

    stats = asyncio.run(pipeline.run(resume=args.resume))

    # Exit code: 0 if any saves, 1 if none
    sys.exit(0 if stats.get("saved", 0) > 0 else 1)


if __name__ == "__main__":
    main()
