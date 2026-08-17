"""Designer module — run-end rubric-refinement caller.

Multi-Agent Execution Plan Slice 3.5
(``plans/multi-agent-execution-plan.md`` §3.5).

The substrate already shipped:

- :func:`market_intelligence.design_market_intelligence.propose_rubric_refinements`
  is the pure function that turns a per-principle feedback marker
  distribution + the brief's current rubric into proposed
  ``RUBRIC_REFINE`` hunks.
- :class:`designer.recruiter_annotations.PrincipleFeedbackStore` owns
  the per-principle feedback log (``annotations.sqlite3`` co-located in
  the per-state-dir SQLite file) and exposes the
  :meth:`feedback_marker_distribution` rollup the proposer consumes.

Slice 3.5 adds the **caller** that wires those two together at the
end of a Designer run, persists the proposed hunks to a deterministic
location under the state-dir, and exposes a loader for the reflection
pipeline (``market_intelligence/reflection.py``) to surface them
alongside the brief-recommendations-derived hunks.

Why a separate module rather than methods on the orchestrator:

- The Slice-1 ``designer/session_orchestrator.py`` is a deliberate
  stub. Keeping the run-end hook as a small, importable module lets
  the reflection-pipeline side load the same persisted artifact
  without taking a hard dependency on the orchestrator (which would
  be circular once the orchestrator's body lands in Slices 5+).
- Tests can exercise the compute / persist / load round-trip without
  spawning the orchestrator subprocess.

Failure posture: every public function tolerates missing inputs
(absent annotations DB, brief without ``design_rubric``, brief with
no calibration exemplars) by returning an empty hunk list and writing
the empty list to the persistence path. This preserves the contract
that "no recruiter feedback" → "no proposed hunks" without crashing
the run-end hook. An empty file still surfaces in the loader as
``[]`` so the reflection pipeline never confuses "missing wiring"
with "wiring ran but no proposals."
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from designer.recruiter_annotations import PrincipleFeedbackStore
from designer.strategy import _dominant_discipline
from market_intelligence.design_market_intelligence import (
    RubricRefineHunk,
    propose_rubric_refinements,
)


# Filename for the per-run persisted hunks under the Designer state-dir.
# Co-located with ``annotations.sqlite3`` so the state-dir carries one
# coherent surface for "what did the recruiter say + what would Cloris
# propose because of it." The reflection pipeline loads from this path
# (see ``market_intelligence/reflection.py``).
PROPOSED_HUNKS_FILENAME = "proposed_rubric_refinement_hunks.json"

# Filename for the per-run annotations SQLite. Mirrors the path the
# Slice-7 recruiter-annotation API writes to (`<state_dir>/annotations.sqlite3`)
# and the smoke test exercises at
# ``tests/test_designer_end_to_end_smoke.py:255``.
ANNOTATIONS_DB_FILENAME = "annotations.sqlite3"


def proposed_rubric_refinement_hunks_path(state_dir: Path) -> Path:
    """Canonical persistence path for the per-run proposed hunks."""

    return Path(state_dir) / PROPOSED_HUNKS_FILENAME


def annotations_db_path(state_dir: Path) -> Path:
    """Canonical path for the per-run recruiter-annotations SQLite."""

    return Path(state_dir) / ANNOTATIONS_DB_FILENAME


def compute_designer_rubric_refinement_hunks(
    *,
    brief: dict[str, Any],
    feedback_distribution: dict[str, dict[str, int]],
) -> list[RubricRefineHunk]:
    """Glue ``_dominant_discipline`` + ``propose_rubric_refinements``.

    Returns ``[]`` for any of the following degenerate inputs (per the
    spec's failure posture — no recruiter feedback yet, no rubric on
    the brief, or no discipline-tagged calibration exemplars):

    - ``brief`` has no ``design_rubric`` (non-Designer brief or
      pre-Slice-4 brief).
    - ``brief.design_rubric`` carries no
      ``calibration_exemplars[*].discipline`` tags
      (``_dominant_discipline`` returns the empty string).
    - ``feedback_distribution`` is empty (no markers recorded yet).
    - All principles' ``useful - off_rubric`` deltas fall under
      :data:`market_intelligence.design_market_intelligence.DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT`.
    """

    rubric = brief.get("design_rubric")
    if not isinstance(rubric, dict):
        return []
    discipline = _dominant_discipline(brief)
    if not discipline:
        return []
    if not feedback_distribution:
        return []
    return propose_rubric_refinements(
        feedback_marker_distribution=feedback_distribution,
        discipline=discipline,
        current_rubric=rubric,
    )


def persist_designer_rubric_refinement_hunks(
    *,
    state_dir: Path,
    hunks: list[RubricRefineHunk],
) -> Path:
    """Write the proposed hunks to ``<state_dir>/proposed_rubric_refinement_hunks.json``.

    Always writes (even an empty list) so a missing file means "the
    run-end hook never ran" and an empty list means "the hook ran and
    nothing crossed threshold." The reflection pipeline distinguishes
    these cases — the first falls back silently, the second is honest
    "no proposals this cycle."

    Returns the path written.
    """

    path = proposed_rubric_refinement_hunks_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(h) for h in hunks]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_designer_rubric_refinement_hunks(
    state_dir: Path,
) -> list[RubricRefineHunk]:
    """Load persisted hunks from the state-dir.

    Returns ``[]`` for any of: missing file, malformed JSON,
    non-list payload, or any individual hunk missing required fields.
    The reflection pipeline (
    ``market_intelligence/reflection.py:reflection_phase_propose``)
    treats an empty result as "nothing to surface" and never raises.
    """

    path = proposed_rubric_refinement_hunks_path(state_dir)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[RubricRefineHunk] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                RubricRefineHunk(
                    label=str(entry["label"]),
                    section=str(entry["section"]),
                    kind=str(entry["kind"]),
                    before=str(entry["before"]),
                    after=str(entry["after"]),
                    rationale=str(entry["rationale"]),
                )
            )
        except (KeyError, TypeError):
            continue
    return out


def run_end_designer_rubric_refinement(
    *,
    brief_path: Path,
    state_dir: Path,
) -> list[RubricRefineHunk]:
    """End-of-run hook the Designer session orchestrator invokes.

    Loads the brief, opens the per-run feedback store if it exists,
    rolls up the per-principle marker distribution, computes proposed
    rubric refinements, and persists them.

    Tolerates missing or unreadable inputs by writing an empty hunk
    list — the run does not fail because the recruiter happened not to
    leave any markers, and the reflection pipeline still gets a
    well-formed (empty) artifact to load.

    Returns the list of computed hunks (also persisted as a side
    effect) so callers and tests can inspect the result without
    re-reading the file.
    """

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    try:
        brief = json.loads(Path(brief_path).read_text())
    except (OSError, json.JSONDecodeError):
        persist_designer_rubric_refinement_hunks(state_dir=state_dir, hunks=[])
        return []

    db_path = annotations_db_path(state_dir)
    if not db_path.exists():
        # Recruiter never opened the workspace surface for this run;
        # no markers exist. Persist an empty list so the loader's
        # "file present, empty list" case is honest.
        persist_designer_rubric_refinement_hunks(state_dir=state_dir, hunks=[])
        return []

    store = PrincipleFeedbackStore(db_path)
    feedback_distribution = store.feedback_marker_distribution()

    hunks = compute_designer_rubric_refinement_hunks(
        brief=brief if isinstance(brief, dict) else {},
        feedback_distribution=feedback_distribution,
    )
    persist_designer_rubric_refinement_hunks(state_dir=state_dir, hunks=hunks)
    return hunks
