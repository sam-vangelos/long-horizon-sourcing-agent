#!/usr/bin/env python3
"""Generate a draft next-version sourcing brief from a structured run report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shared.brief_iteration import iterate_brief_draft


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a draft revised brief from a structured run report"
    )
    parser.add_argument("--brief", required=True, help="Path to the source brief JSON")
    parser.add_argument("--report", default=None, help="Path to run-report.json")
    parser.add_argument(
        "--search-memory",
        default=None,
        help="Optional path to search_memory-<project>.json",
    )
    parser.add_argument(
        "--final-judgments",
        default=None,
        help="Optional path to final_judgments.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for rationale artifacts and default report resolution",
    )
    args = parser.parse_args()

    try:
        result = iterate_brief_draft(
            brief_path=args.brief,
            report_path=args.report,
            search_memory_path=args.search_memory,
            final_judgments_path=args.final_judgments,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"[error] Brief iteration failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Draft brief generated.")
    print(f"  Draft brief: {Path(result.draft_brief_path)}")
    print(f"  Rationale:   {Path(result.rationale_path)}")
    if result.warnings:
        print("  Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")


if __name__ == "__main__":
    main()
