#!/usr/bin/env python3
"""Validate the final LinkedIn live-evidence bucket for SQK completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from linkedin.empirical_register import (  # noqa: E402
    final_linkedin_live_bucket_payload_template,
    validate_final_linkedin_live_bucket,
)
from linkedin.matching_contract import MATCHING_CONTRACT_ARTIFACT_PATH  # noqa: E402


EXIT_INVALID = 1
EXIT_PENDING = 2


def _persist_matching_contract(contract: dict[str, Any], path: Path) -> None:
    """Write the verified matching contract artifact. Operator action only —

    never called unless --persist-contract is explicitly passed, so a test run
    (or an incomplete/invalid payload) never fabricates or overwrites the
    checked-in config/ artifact.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_payload(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate supplied final LinkedIn live evidence. Exit 0 only when "
            "all final live gates are complete."
        )
    )
    parser.add_argument(
        "payload",
        nargs="?",
        help="Path to final live-bucket JSON evidence, or '-' for stdin.",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Print the canonical JSON payload template and exit.",
    )
    parser.add_argument(
        "--persist-contract",
        action="store_true",
        help=(
            "When validation is complete, write the verified matching contract "
            "to --contract-out. Off by default so incomplete runs and tests "
            "never fabricate the artifact."
        ),
    )
    parser.add_argument(
        "--contract-out",
        default=str(MATCHING_CONTRACT_ARTIFACT_PATH),
        help=(
            "Path to write the verified matching contract when --persist-contract "
            f"is set (default: {MATCHING_CONTRACT_ARTIFACT_PATH})."
        ),
    )
    args = parser.parse_args(argv)

    if args.template:
        print(
            json.dumps(
                final_linkedin_live_bucket_payload_template(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.payload:
        parser.error("payload is required unless --template is used")

    try:
        payload = _read_payload(args.payload)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "linkedin.final_live_bucket.v1",
                    "status": "invalid",
                    "pending_gates": [],
                    "errors": [str(exc)],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return EXIT_INVALID

    report = validate_final_linkedin_live_bucket(payload)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.status == "complete":
        if args.persist_contract and report.matching_contract:
            _persist_matching_contract(dict(report.matching_contract), Path(args.contract_out))
        return 0
    if report.status == "pending":
        return EXIT_PENDING
    return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
