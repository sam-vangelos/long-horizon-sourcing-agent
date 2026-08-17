"""Import-boundary locks for the sunset modules (Designer + Exec Search).

A5 (spec §A5, Sam D-8) archives Designer and Exec Search out of runtime
wiring. Its *minimum* requirement — that ``shared.judger`` not drag those
packages in at module-import time — is already satisfied today: the only
``exec_search`` imports in ``shared/judger.py`` sit inside a function behind
a ``dossier_mode`` guard, and it imports ``designer`` not at all.

That property is currently true by accident. These locks make it mechanical,
so the full archive starts from a guarded position rather than re-deriving
the inventory.

Each check runs in a FRESH interpreter via subprocess. An in-process
``sys.modules`` assertion would be order-dependent: any earlier test in the
same session that imports ``designer`` or ``exec_search`` would pollute the
module table and make the result depend on collection order rather than on
the import graph.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_SUNSET_ROOTS = ("designer", "exec_search")

_PROBE = """
import sys
import {module}
leaked = sorted(
    name for name in sys.modules
    if name.split(".")[0] in {roots!r}
)
print(",".join(leaked))
"""


def _sunset_modules_pulled_in_by(module: str) -> list[str]:
    """Import ``module`` in a clean interpreter; return sunset packages loaded."""

    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, roots=_SUNSET_ROOTS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"probe failed to import {module}:\n{result.stderr}"
    )
    printed = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return [name for name in printed.split(",") if name]


def test_importing_judger_does_not_pull_sunset_packages():
    """shared.judger must not import Designer or Exec Search at module scope.

    The exec_search evidence-assembly imports live inside
    ``shared/judger.py``'s dossier-mode branch and must STAY function-local:
    hoisting them to module scope would couple every judgment path — LinkedIn
    and GitHub included — to a package being archived.
    """

    assert _sunset_modules_pulled_in_by("shared.judger") == []


def test_importing_linkedin_orchestrator_does_not_pull_sunset_packages():
    """The live LinkedIn run path must not drag the sunset packages in."""

    assert _sunset_modules_pulled_in_by("linkedin.orchestrator") == []


@pytest.mark.parametrize("root", _SUNSET_ROOTS)
def test_probe_detects_a_real_sunset_import(root: str):
    """Prove the probe can actually SEE a sunset import.

    Without this, both locks above would pass just as happily against a
    broken probe that always returns an empty list.
    """

    assert _sunset_modules_pulled_in_by(root) != []
