"""Multi-Agent Execution Plan Slice 3.5 — Designer per-principle wiring.

Pins the wiring contract between three previously-uncoupled surfaces:

1. :class:`designer.recruiter_annotations.PrincipleFeedbackStore` —
   per-principle recruiter feedback marker store
   (``designer/recruiter_annotations.py:313`` ``feedback_marker_distribution``).
2. :func:`market_intelligence.design_market_intelligence.propose_rubric_refinements`
   — pure function that turns the distribution into proposed
   ``RUBRIC_REFINE`` hunks
   (``market_intelligence/design_market_intelligence.py:212``).
3. :func:`market_intelligence.reflection.reflection_phase_propose` —
   the propose-phase hunks list the recruiter reviews at Gate 2.

The slice ships:

- :mod:`designer.run_end` — the run-end caller (compute → persist).
- :func:`designer.session_orchestrator.main` — invokes the run-end
  caller after the Slice-1 stub body.
- :func:`market_intelligence.reflection._designer_rubric_refine_propose_hunks`
  — loads the persisted hunks and projects them onto the propose-phase
  hunk dict shape so they surface alongside the
  brief-recommendations-derived hunks.

Tests below cover the chain end-to-end with deterministic fixtures.
No real LLM calls; no real reflection_phase_propose run (that would
require a planner result, market identity, and evidence batches —
out of scope for what Slice 3.5 actually wires).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from designer.recruiter_annotations import PrincipleFeedbackStore
from designer.run_end import (
    PROPOSED_HUNKS_FILENAME,
    annotations_db_path,
    compute_designer_rubric_refinement_hunks,
    load_designer_rubric_refinement_hunks,
    persist_designer_rubric_refinement_hunks,
    proposed_rubric_refinement_hunks_path,
    run_end_designer_rubric_refinement,
)
from market_intelligence.design_market_intelligence import (
    DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT,
    RubricRefineHunk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _designer_brief(*, with_rubric: bool = True, discipline: str = "product") -> dict:
    """Synthetic Designer brief carrying the keys the run-end hook reads.

    The brief schema bits the hook actually consumes:
    - ``target_modules`` — must include ``"designer"`` for the
      reflection helper to load the persisted hunks.
    - ``design_rubric.calibration_exemplars[*].discipline`` — drives
      ``_dominant_discipline``; without a tagged exemplar the
      ``compute_designer_rubric_refinement_hunks`` helper short-
      circuits to ``[]``.
    - ``design_rubric.discipline_weight_overrides`` — read by
      ``propose_rubric_refinements`` to compute current weights.
    """

    brief: dict[str, Any] = {
        "id": "designer-test-brief",
        "role_title": "Senior product designer",
        "target_modules": ["designer"],
    }
    if with_rubric:
        brief["design_rubric"] = {
            "calibration_exemplars": [
                {"discipline": discipline, "name": "Sample exemplar"},
            ],
            "discipline_weight_overrides": {discipline: {}},
        }
    return brief


def _write_brief(tmp_path: Path, brief: dict) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief))
    return path


# ---------------------------------------------------------------------------
# compute_designer_rubric_refinement_hunks
# ---------------------------------------------------------------------------


def test_compute_returns_empty_when_brief_has_no_rubric() -> None:
    """No rubric on the brief — Slice 1/2 brief or non-Designer brief."""

    brief = _designer_brief(with_rubric=False)
    feedback_distribution = {
        "Visual hierarchy": {"useful_guidance": 100, "off_rubric": 0}
    }
    hunks = compute_designer_rubric_refinement_hunks(
        brief=brief, feedback_distribution=feedback_distribution
    )
    assert hunks == []


def test_compute_returns_empty_when_no_dominant_discipline() -> None:
    """Brief carries a rubric but no discipline-tagged exemplars."""

    brief = _designer_brief()
    brief["design_rubric"]["calibration_exemplars"] = [{"name": "no-discipline"}]
    feedback_distribution = {
        "Visual hierarchy": {"useful_guidance": 100, "off_rubric": 0}
    }
    hunks = compute_designer_rubric_refinement_hunks(
        brief=brief, feedback_distribution=feedback_distribution
    )
    assert hunks == []


def test_compute_returns_empty_when_no_feedback_yet() -> None:
    """Brief is well-shaped but no recruiter has touched the workspace."""

    brief = _designer_brief()
    hunks = compute_designer_rubric_refinement_hunks(
        brief=brief, feedback_distribution={}
    )
    assert hunks == []


def test_compute_proposes_weight_up_at_threshold() -> None:
    brief = _designer_brief(discipline="product")
    feedback_distribution = {
        "Visual hierarchy": {
            "useful_guidance": DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT,
            "off_rubric": 0,
        }
    }
    hunks = compute_designer_rubric_refinement_hunks(
        brief=brief, feedback_distribution=feedback_distribution
    )
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.kind == "rubric_refine"
    assert hunk.section == "design_rubric.discipline_weight_overrides"
    assert "higher" in hunk.label
    assert "product" in hunk.label
    assert "1.3" in hunk.after  # default +0.3 step from baseline 1.0


def test_compute_proposes_weight_down_at_threshold() -> None:
    brief = _designer_brief(discipline="product")
    brief["design_rubric"]["discipline_weight_overrides"]["product"] = {
        "Typographic refinement": 1.0
    }
    feedback_distribution = {
        "Typographic refinement": {
            "useful_guidance": 0,
            "off_rubric": DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT,
        }
    }
    hunks = compute_designer_rubric_refinement_hunks(
        brief=brief, feedback_distribution=feedback_distribution
    )
    assert len(hunks) == 1
    assert "lower" in hunks[0].label
    assert "0.7" in hunks[0].after


# ---------------------------------------------------------------------------
# persist + load round-trip
# ---------------------------------------------------------------------------


def test_persist_writes_canonical_filename(tmp_path: Path) -> None:
    written = persist_designer_rubric_refinement_hunks(
        state_dir=tmp_path, hunks=[]
    )
    assert written == tmp_path / PROPOSED_HUNKS_FILENAME
    assert written.exists()


def test_persist_writes_empty_list_as_empty_json_array(tmp_path: Path) -> None:
    """Empty list → empty JSON array. Distinct from "file missing"
    (which means "the run-end hook never ran")."""

    persist_designer_rubric_refinement_hunks(state_dir=tmp_path, hunks=[])
    payload = json.loads((tmp_path / PROPOSED_HUNKS_FILENAME).read_text())
    assert payload == []


def test_persist_load_round_trips_hunks(tmp_path: Path) -> None:
    hunk = RubricRefineHunk(
        label="Weight Visual hierarchy higher for product",
        section="design_rubric.discipline_weight_overrides",
        kind="rubric_refine",
        before="product: {Visual hierarchy: 1.0}",
        after="product: {Visual hierarchy: 1.3}",
        rationale="Recruiters consistently marked Visual hierarchy useful.",
    )
    persist_designer_rubric_refinement_hunks(state_dir=tmp_path, hunks=[hunk])
    loaded = load_designer_rubric_refinement_hunks(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == hunk


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_designer_rubric_refinement_hunks(tmp_path) == []


def test_load_returns_empty_for_malformed_json(tmp_path: Path) -> None:
    proposed_rubric_refinement_hunks_path(tmp_path).write_text("not json {{")
    assert load_designer_rubric_refinement_hunks(tmp_path) == []


def test_load_skips_entries_missing_required_fields(tmp_path: Path) -> None:
    """A partial hunk row drops out; well-formed siblings still load."""

    proposed_rubric_refinement_hunks_path(tmp_path).write_text(
        json.dumps(
            [
                {"label": "incomplete"},
                {
                    "label": "complete",
                    "section": "design_rubric.discipline_weight_overrides",
                    "kind": "rubric_refine",
                    "before": "x",
                    "after": "y",
                    "rationale": "z",
                },
            ]
        )
    )
    loaded = load_designer_rubric_refinement_hunks(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].label == "complete"


# ---------------------------------------------------------------------------
# run_end_designer_rubric_refinement — full integration with the store
# ---------------------------------------------------------------------------


def test_run_end_with_no_annotations_db_persists_empty_list(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _designer_brief())
    state_dir = tmp_path / "state"

    hunks = run_end_designer_rubric_refinement(
        brief_path=brief_path, state_dir=state_dir
    )

    assert hunks == []
    persisted = proposed_rubric_refinement_hunks_path(state_dir)
    assert persisted.exists()
    assert json.loads(persisted.read_text()) == []


def test_run_end_with_unreadable_brief_persists_empty_list(tmp_path: Path) -> None:
    bogus_brief = tmp_path / "missing.json"
    state_dir = tmp_path / "state"

    hunks = run_end_designer_rubric_refinement(
        brief_path=bogus_brief, state_dir=state_dir
    )

    assert hunks == []
    assert proposed_rubric_refinement_hunks_path(state_dir).exists()


def test_run_end_persists_proposed_hunks_from_feedback_store(tmp_path: Path) -> None:
    """End-to-end: stage the annotations DB, run the hook, read the JSON."""

    brief_path = _write_brief(tmp_path, _designer_brief(discipline="product"))
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    store = PrincipleFeedbackStore(annotations_db_path(state_dir))
    for _ in range(DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT):
        store.record(
            candidate_identity_key="behance:joe",
            principle_name="Visual hierarchy",
            marker="useful_guidance",
        )

    hunks = run_end_designer_rubric_refinement(
        brief_path=brief_path, state_dir=state_dir
    )

    assert len(hunks) == 1
    assert hunks[0].kind == "rubric_refine"
    persisted = json.loads(
        proposed_rubric_refinement_hunks_path(state_dir).read_text()
    )
    assert len(persisted) == 1
    assert persisted[0]["section"] == "design_rubric.discipline_weight_overrides"


# ---------------------------------------------------------------------------
# session_orchestrator main() — Slice 3.5 wires the run-end hook
# ---------------------------------------------------------------------------


def test_session_orchestrator_main_invokes_run_end_hook(tmp_path: Path) -> None:
    """The Slice-1 stub body still runs, AND the Slice 3.5 hook fires."""

    brief_path = _write_brief(tmp_path, _designer_brief(discipline="product"))
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    store = PrincipleFeedbackStore(annotations_db_path(state_dir))
    for _ in range(DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT):
        store.record(
            candidate_identity_key="behance:joe",
            principle_name="Visual hierarchy",
            marker="useful_guidance",
        )

    from designer import session_orchestrator

    rc = session_orchestrator.main(
        ["--brief", str(brief_path), "--state-dir", str(state_dir)]
    )
    assert rc == 0

    persisted = json.loads(
        proposed_rubric_refinement_hunks_path(state_dir).read_text()
    )
    assert len(persisted) == 1
    assert persisted[0]["section"] == "design_rubric.discipline_weight_overrides"


def test_session_orchestrator_main_succeeds_when_hook_persists_empty(
    tmp_path: Path,
) -> None:
    """Slice-1 baseline: no annotations DB, no proposals, exit 0 still."""

    brief_path = _write_brief(tmp_path, _designer_brief())
    state_dir = tmp_path / "state"

    from designer import session_orchestrator

    rc = session_orchestrator.main(
        ["--brief", str(brief_path), "--state-dir", str(state_dir)]
    )
    assert rc == 0
    assert proposed_rubric_refinement_hunks_path(state_dir).exists()
    assert json.loads(
        proposed_rubric_refinement_hunks_path(state_dir).read_text()
    ) == []


def test_session_orchestrator_main_subprocess_writes_artifact(
    tmp_path: Path,
) -> None:
    """Belt-and-braces: spawn the orchestrator the way `cloris.worker`
    spawns it (via `python -m designer.session_orchestrator`) and
    confirm the run-end artifact still lands. Catches PYTHONPATH /
    entrypoint regressions a pure-import test would miss.
    """

    brief_path = _write_brief(tmp_path, _designer_brief())
    state_dir = tmp_path / "state"
    repo_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "designer.session_orchestrator",
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert proposed_rubric_refinement_hunks_path(state_dir).exists()


# ---------------------------------------------------------------------------
# Reflection-pipeline projection — Slice 3.5's "surface in reflection"
# ---------------------------------------------------------------------------


def _stub_designer_state_dir(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Force the reflection helper to look up `target` as the Designer
    state dir for any brief, so tmp_path-rooted tests don't pollute
    the repo's real `output/state/designer/` tree."""

    monkeypatch.setattr(
        "shared.output_paths.resolve_designer_state_dir",
        lambda **_kwargs: target,
    )


def test_reflection_helper_returns_empty_for_non_designer_brief(
    tmp_path: Path,
) -> None:
    from market_intelligence.reflection import (
        _designer_rubric_refine_propose_hunks,
    )

    brief_raw = {"target_modules": ["linkedin"]}
    out = _designer_rubric_refine_propose_hunks(
        brief_raw=brief_raw, brief_path=tmp_path / "brief.json"
    )
    assert out == []


def test_reflection_helper_returns_empty_when_target_modules_missing(
    tmp_path: Path,
) -> None:
    from market_intelligence.reflection import (
        _designer_rubric_refine_propose_hunks,
    )

    out = _designer_rubric_refine_propose_hunks(
        brief_raw={}, brief_path=tmp_path / "brief.json"
    )
    assert out == []


def test_reflection_helper_returns_empty_when_state_dir_has_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from market_intelligence.reflection import (
        _designer_rubric_refine_propose_hunks,
    )

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _stub_designer_state_dir(monkeypatch, state_dir)

    out = _designer_rubric_refine_propose_hunks(
        brief_raw={"target_modules": ["designer"]},
        brief_path=tmp_path / "brief.json",
    )
    assert out == []


def test_reflection_helper_projects_persisted_hunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: orchestrator persists hunks → reflection helper
    surfaces them in the propose-phase hunk dict shape that
    HunkCard.svelte already renders."""

    from market_intelligence.reflection import (
        _designer_rubric_refine_propose_hunks,
    )

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _stub_designer_state_dir(monkeypatch, state_dir)

    persist_designer_rubric_refinement_hunks(
        state_dir=state_dir,
        hunks=[
            RubricRefineHunk(
                label="Weight Visual hierarchy higher for product",
                section="design_rubric.discipline_weight_overrides",
                kind="rubric_refine",
                before="product: {Visual hierarchy: 1.0}",
                after="product: {Visual hierarchy: 1.3}",
                rationale="Recruiters consistently marked it useful.",
            ),
            RubricRefineHunk(
                label="Weight Color system coherence lower for product",
                section="design_rubric.discipline_weight_overrides",
                kind="rubric_refine",
                before="product: {Color system coherence: 1.0}",
                after="product: {Color system coherence: 0.7}",
                rationale="Recruiters consistently marked it off-rubric.",
            ),
        ],
    )

    hunks = _designer_rubric_refine_propose_hunks(
        brief_raw={"target_modules": ["designer"]},
        brief_path=tmp_path / "brief.json",
    )
    assert len(hunks) == 2
    first, second = hunks
    assert first["kind"] == "rubric_refine"
    assert first["section"] == "design_rubric.discipline_weight_overrides"
    assert first["hunk_id"] == "rubric-refine-1"
    assert first["default_approved"] is False
    assert "higher" in first["label"]
    assert second["hunk_id"] == "rubric-refine-2"
    assert "lower" in second["label"]
    # Required wire fields the propose-block reader (
    # cloris/frontend/src/lib/reflection/types.ts:normalizeHunk)
    # expects to be present and well-typed.
    for hunk in hunks:
        for key in (
            "hunk_id",
            "section",
            "kind",
            "label",
            "before",
            "after",
            "rationale",
            "confidence",
            "default_approved",
            "target_field",
        ):
            assert key in hunk
        assert isinstance(hunk["confidence"], float)


def test_reflection_helper_returns_empty_when_resolve_state_dir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brief with ``target_modules: ["designer"]`` but a malformed
    body (no ``id`` / no ``role_title``) can make
    ``resolve_designer_state_dir`` raise on the loader's behalf;
    the helper should swallow and return ``[]`` rather than
    propagate the exception out of ``reflection_phase_propose``."""

    from market_intelligence.reflection import (
        _designer_rubric_refine_propose_hunks,
    )

    def boom(**_kwargs: Any) -> Path:
        raise RuntimeError("cannot resolve designer state dir")

    monkeypatch.setattr("shared.output_paths.resolve_designer_state_dir", boom)

    out = _designer_rubric_refine_propose_hunks(
        brief_raw={"target_modules": ["designer"]},
        brief_path=tmp_path / "brief.json",
    )
    assert out == []
