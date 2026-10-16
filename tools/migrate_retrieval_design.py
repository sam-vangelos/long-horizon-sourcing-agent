"""Seed an explicit retrieval_design block into an existing brief."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.retrieval_design import retrieval_design_from_payload
from shared.storage import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed retrieval_design into a brief JSON file.")
    parser.add_argument("--brief", required=True, help="Path to the source brief JSON.")
    parser.add_argument("--output", help="Optional output path. Defaults to <brief>.retrieval.json")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the source brief in place instead of writing a sibling file.",
    )
    args = parser.parse_args()

    brief_path = Path(args.brief)
    if not brief_path.exists():
        raise FileNotFoundError(f"brief not found: {brief_path}")

    raw = read_json(brief_path)
    if not isinstance(raw, dict):
        raise ValueError("brief payload must be a JSON object")

    raw["retrieval_design"] = retrieval_design_from_payload(
        raw.get("retrieval_design"),
        legacy_search_priorities=raw.get("search_priorities", []),
        legacy_additional_search_terms=raw.get("additional_search_terms", []),
        role_title=raw.get("role_title", ""),
    ).to_dict()

    output_path = (
        brief_path
        if args.in_place
        else Path(args.output)
        if args.output
        else brief_path.with_name(f"{brief_path.stem}.retrieval{brief_path.suffix}")
    )
    write_json(output_path, raw)
    print(output_path)


if __name__ == "__main__":
    main()
