#!/usr/bin/env python3
"""Browser-free rejudging of previously extracted LinkedIn snippets."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from shared.console_tee import enable_console_tee
from shared.output_paths import resolve_linkedin_state_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rejudge existing LinkedIn snippet JSONL without browser access"
    )
    parser.add_argument("--brief", required=True, help="Path to the sourcing brief JSON")
    parser.add_argument(
        "--rejudge-from",
        required=True,
        help="Existing snippet JSONL to rejudge",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Mutable brief-scoped state directory",
    )
    parser.add_argument("--output-dir", default=None, help="Deprecated alias for --state-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    brief_path = Path(args.brief)
    snippets_path = Path(args.rejudge_from)
    if not brief_path.is_file():
        raise SystemExit(f"Brief file not found: {brief_path}")
    if not snippets_path.is_file():
        raise SystemExit(f"Snippet file not found: {snippets_path}")

    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=args.state_dir or args.output_dir,
    )
    enable_console_tee(state_dir)

    from linkedin.orchestrator import Pipeline

    pipeline = Pipeline(
        brief_path=str(brief_path),
        output_dir=str(state_dir),
    )
    asyncio.run(pipeline.rejudge_from_file(str(snippets_path)))


if __name__ == "__main__":
    main()
