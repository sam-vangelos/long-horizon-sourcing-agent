"""Stand-ins for briefs that tests were written against and that left the tree.

Fifty tests read ``config/brief-head-ai-lab-nyc-v2.json``, a strict-seniority
bank search for a Head of Applied AI Lab. That file is in no commit reachable
from HEAD, so every one of them skipped on every machine (class E of
``plans/test-theater-audit.md``). They now run against the tracked dev fixture
``config/head-of-applied-ai-fixture/brief.json``, given the identity and the few
fields the tests' own fake run data is keyed to. Nothing here restores the old
brief: the overrides are synthetic, and each says which assertions need it.
"""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TRACKED_FIXTURE = ROOT / "config" / "head-of-applied-ai-fixture" / "brief.json"

# The project id the tests' runtime rows, state dirs and artifact names carry.
HEAD_AI_LAB_PROJECT_ID = "9000000001"

_OVERRIDES: dict[str, object] = {
    # Identity: the fake run reports are stamped brief_name "head-ai-lab",
    # role "Head of Applied AI Lab", version "2.1". A snapshot only adopts a
    # report that belongs to its brief.
    "id": "head-ai-lab",
    "role_title": "Head of Applied AI Lab",
    "version": "2.1",
    # Live briefs carry the project flat as well as under source_config; the
    # loader reads the flat field.
    "linkedin_project": "Head of Applied AI Lab",
    "linkedin_project_id": HEAD_AI_LAB_PROJECT_ID,
    # Locked fields the iteration tests assert are preserved.
    "market_density": "moderate",
    "facial_calibration": {"expected_yes_rate_low": 0.25, "expected_yes_rate_high": 0.5},
    # A brief without a selectivity posture logs a warning, and one test
    # asserts a clean log.
    "engagement_context": {"selectivity_posture": "selective"},
    # The brief these tests were written for was a strict-seniority bank search
    # (shared/strict_seniority.py: twelve or more years, an Executive Director
    # trigger, a bank). The guardrail tests need that character.
    "minimum_years_experience": 15,
    "intake_notes": (
        "A bank's applied-AI lab head: Executive Director scope, or clearly "
        "ED-analogous, one layer below broad enterprise executives. Hands-on "
        "lab-head scope, not a portfolio executive."
    ),
}


@lru_cache(maxsize=1)
def head_ai_lab_stand_in() -> Path:
    """Path to the stand-in brief, written once per process under a temp dir.

    It keeps the old file's name because tests copy it as ``SOURCE_BRIEF.name``.
    """
    payload = json.loads(_TRACKED_FIXTURE.read_text(encoding="utf-8"))
    payload.update(_OVERRIDES)
    linkedin = payload.setdefault("source_config", {}).setdefault("linkedin", {})
    linkedin["project_id"] = HEAD_AI_LAB_PROJECT_ID
    linkedin["project_name"] = "Head of Applied AI Lab"
    payload.setdefault("depth_distinction", {}).setdefault(
        "edge_case_guidance", "Weigh shipped applied-AI ownership over title."
    )
    directory = Path(tempfile.mkdtemp(prefix="cloris-brief-stand-in-"))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    path = directory / "brief-head-ai-lab-nyc-v2.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
