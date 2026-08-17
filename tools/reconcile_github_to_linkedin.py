"""Retired CLI for the legacy GitHub→LinkedIn reconciliation path.

This entry point used to drive
``linkedin.reconciliation.LinkedInReconciliationService``, which emits
a non-canonical action taxonomy and does not run the LinkedIn brief's
holistic fit judge. That behavior violates the non-negotiable rules
defined in ``GitHub-LinkedIn-Reconciliation-Source-of-Truth.md`` and
is therefore no longer runnable.

Use the canonical CLI instead:

    python3 tools/run_recruiter_identity_resolver.py --help

See the "Canonical Implementation" section of
``GitHub-LinkedIn-Reconciliation-Source-of-Truth.md`` for background.
"""

from __future__ import annotations

import sys


RETIRED_MESSAGE = (
    "tools/reconcile_github_to_linkedin.py is retired.\n"
    "\n"
    "The legacy LinkedInReconciliationService predates the canonical\n"
    "reconciliation contract in\n"
    "GitHub-LinkedIn-Reconciliation-Source-of-Truth.md and must not be\n"
    "used to produce recruiter-facing artifacts.\n"
    "\n"
    "Use the canonical entry point instead:\n"
    "    python3 tools/run_recruiter_identity_resolver.py --help\n"
)


def main() -> int:
    sys.stderr.write(RETIRED_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
