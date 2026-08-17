"""Multi-Agent Execution Plan Slice 3.4 — reflection-pipeline integration.

Pins the wiring contract between four previously-uncoupled surfaces:

1. :func:`shared.runtime_state.calibration.aggregate_calibration_markers`
   (Slice 3.1) — pure read of ``judgment_accuracy`` rows from a per-
   source ``runtime_state.sqlite3``.
2. :func:`market_intelligence.calibration_thresholds.select_eligible_areas`
   (Slice 3.2) — gates the rollup on per-area + per-cycle thresholds.
3. :func:`market_intelligence.calibration_to_brief.translate_eligible_areas`
   (Slice 3.3) — projects eligible areas onto V2 brief patches per
   three pattern rules.
4. :func:`market_intelligence.reflection.reflection_phase_propose` —
   the propose-phase hunks list the recruiter reviews at Gate 2.

The slice ships:

- :func:`market_intelligence.reflection._calibration_propose_hunks` —
  composes 1 → 2 → 3 → propose-phase hunk dict shape so calibration
  patches surface alongside brief-recommendations-derived hunks.

The slice card's "end-to-end" test requirement is satisfied by:

- :func:`test_helper_surfaces_three_pattern_hunks_from_seeded_markers` —
  fixture multi-run brief with marker history → helper returns the
  expected per-pattern hunks in the pinned shape. This is the
  load-bearing test; it exercises the aggregator + threshold layer +
  translator + state-dir resolution against a real seeded SQLite.
- :func:`test_helper_returns_empty_when_no_state_dirs_match_brief_id`
  and :func:`test_helper_returns_empty_when_brief_has_no_id` — the
  defensive failure-mode coverage (helper composes cleanly when the
  brief is too new or has no id, since calibration is a passive
  observer that should never crash the reflection phase).
- :func:`test_calibration_hunks_merge_alongside_brief_recommendation_hunks`
  — the slice's actual ask: brief-recommendation hunks + calibration
  hunks share the same Gate-2 list, in deterministic order.

The propose phase itself is NOT exercised end-to-end here — the LLM
synthesis + critic backends + evidence batches + planner result are
out of scope for what Slice 3.4 actually wires. Slice 3.5's parallel
test (:mod:`tests.test_designer_rubric_refinement_wiring`) takes the
same posture for the same reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from market_intelligence.reflection import (
    _build_hunks_from_artifact,
    _calibration_propose_hunks,
)
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_store(state_dir: Path) -> RuntimeStateStore:
    """Construct a runtime-state store at the canonical location.

    The aggregator reads from ``state_dir / "runtime_state.sqlite3"``
    (mirrors the per-source state-dir layout production uses; see
    ``cloris.control_plane.state_dirs_for_brief_id`` for the resolver
    side).
    """

    state_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeStateStore(state_dir / "runtime_state.sqlite3")


def _seed_marker(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    capability_area: str | None,
    confidence: float | None,
    terminal_decision: str,
    judgment_accuracy: str,
) -> int:
    """Walk a single candidate from discovered → full_terminal + stamp marker.

    Mirrors the fixture in
    ``tests/test_calibration_aggregator.py::_seed_saved_candidate``.
    Inlined here (rather than imported) because the aggregator test
    keeps its helpers private and we don't want a cross-test import
    coupling that would force coordination on every fixture tweak.
    """

    runs = store.list_runs(source=source, brief_id=brief_id)
    if runs:
        run_id = int(runs[0]["id"])
    else:
        run_id = store.start_run(
            source=source,
            brief_id=brief_id,
            output_dir=str(store.db_path.parent),
            mode="fresh",
            resume_state={"brief_name": brief_id},
        )

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=f"Candidate {identity_key}",
        profile_url=f"https://example.test/{identity_key}",
    )
    for new_state in (
        "snippet_extracted",
        "facial_started",
        "facial_terminal",
        "full_started",
    ):
        store.set_candidate_state(
            run_id=run_id,
            source=source,
            brief_id=brief_id,
            identity_key=identity_key,
            new_state=new_state,
        )

    full_decision: dict[str, Any] = {
        "decision": terminal_decision,
        "rationale": f"rationale for {identity_key}",
    }
    if confidence is not None:
        full_decision["confidence"] = confidence
    if capability_area is not None:
        full_decision["capability_area"] = capability_area

    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_terminal",
        terminal_decision=terminal_decision,
        terminal_payload={"full_decision": full_decision},
    )

    candidate = store.get_candidate(
        source=source, brief_id=brief_id, identity_key=identity_key
    )
    assert candidate is not None
    candidate_id = int(candidate["id"])
    store.set_candidate_judgment_accuracy(candidate_id, judgment_accuracy)
    return candidate_id


def _stub_state_dirs_for_brief_id(
    monkeypatch: pytest.MonkeyPatch,
    state_dirs: list[tuple[str, Path]],
) -> None:
    """Patch the control-plane resolver the helper imports lazily.

    The helper in ``market_intelligence/reflection.py`` does
    ``from cloris.control_plane import state_dirs_for_brief_id``
    inside the function body. We patch the source-of-truth attribute
    at ``cloris.control_plane`` rather than the helper-local binding
    so the late-import gets the patched callable. Mirrors the
    Slice-3.5 test pattern at
    ``tests/test_designer_rubric_refinement_wiring.py:_stub_designer_state_dir``.
    """

    monkeypatch.setattr(
        "cloris.control_plane.state_dirs_for_brief_id",
        lambda _brief_id: list(state_dirs),
    )


class _StubBrief:
    """Minimal Brief stand-in carrying only the ``id`` field the helper reads.

    The real ``shared.brief_loader.Brief`` carries 30+ fields; the
    calibration helper only ever reads ``brief.id``. A typed stub
    keeps test setup honest about the actual coupling and avoids
    a brittle dependency on the loader's full schema.
    """

    def __init__(self, brief_id: str) -> None:
        self.id = brief_id


# ---------------------------------------------------------------------------
# End-to-end: helper against seeded state DB
# ---------------------------------------------------------------------------


def test_helper_surfaces_three_pattern_hunks_from_seeded_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slice card test plan: fixture multi-run brief with marker history
    → ``_calibration_propose_hunks`` returns one calibration hunk per
    eligible (area, pattern) pair.

    Seeds three capability areas, each crossing the per-area threshold
    (≥5 weighted markers) AND the per-pattern floor (≥3 raw markers
    of the dominant marker class):

    - ``Foundation Models Research``: 5 ``wrong`` markers → non_fit_pattern.
    - ``Inference Optimization``: 4 ``off_rubric`` markers in q4 + SAVE
      (high-confidence saves the recruiter said were on the wrong axis)
      → depth_distinction.
    - ``Frontend Performance``: 6 ``useful`` markers → calibration_examples.

    Asserts: three calibration hunks surface, one per pattern, with
    the wire-shape every Gate-2 hunk consumer (``HunkCard.svelte`` +
    ``buildMergedV2`` in ``cloris/frontend/src/lib/briefDiff.ts``)
    expects.
    """

    state_dir = tmp_path / "linkedin" / "state-key-1"
    store = _make_store(state_dir)

    brief_id = "brief-multi-area"
    fixtures = [
        # Pattern 1: 5 wrong markers in one area (q3 confidence so they
        # don't double-count for the high-confidence weighted bonus).
        ("li-w1", "Foundation Models Research", 0.6, "REJECT", "wrong"),
        ("li-w2", "Foundation Models Research", 0.6, "REJECT", "wrong"),
        ("li-w3", "Foundation Models Research", 0.6, "REJECT", "wrong"),
        ("li-w4", "Foundation Models Research", 0.6, "REJECT", "wrong"),
        ("li-w5", "Foundation Models Research", 0.6, "REJECT", "wrong"),
        # Pattern 2: 4 off_rubric markers on high-confidence SAVE rows.
        ("li-o1", "Inference Optimization", 0.85, "SAVE", "off_rubric"),
        ("li-o2", "Inference Optimization", 0.90, "SAVE", "off_rubric"),
        ("li-o3", "Inference Optimization", 0.80, "SAVE", "off_rubric"),
        ("li-o4", "Inference Optimization", 0.95, "SAVE", "off_rubric"),
        # Pattern 3: 6 useful markers (clears both per-area + per-pattern
        # floors with margin).
        ("li-u1", "Frontend Performance", 0.7, "SAVE", "useful"),
        ("li-u2", "Frontend Performance", 0.7, "SAVE", "useful"),
        ("li-u3", "Frontend Performance", 0.7, "SAVE", "useful"),
        ("li-u4", "Frontend Performance", 0.7, "SAVE", "useful"),
        ("li-u5", "Frontend Performance", 0.7, "SAVE", "useful"),
        ("li-u6", "Frontend Performance", 0.7, "SAVE", "useful"),
    ]
    for identity_key, area, conf, decision, marker in fixtures:
        _seed_marker(
            store,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            capability_area=area,
            confidence=conf,
            terminal_decision=decision,
            judgment_accuracy=marker,
        )

    _stub_state_dirs_for_brief_id(monkeypatch, [("linkedin", state_dir)])

    brief = _StubBrief(brief_id=brief_id)
    brief_raw: dict = {"id": brief_id}

    hunks = _calibration_propose_hunks(
        brief=brief,
        brief_raw=brief_raw,
        brief_path=tmp_path / "brief.json",
    )

    # Three patterns each clearing their per-pattern floor → three hunks.
    # The hunks are ordered by (eligible-area rank, inner-pattern fixed
    # order). Eligible-area rank is by weighted_markers_by_area desc;
    # all three areas tie on weighted count = raw count (no high-confidence
    # bonus on `useful` markers, and the wrong/off_rubric markers in this
    # fixture sit at quartiles where the bonus does/doesn't fire). We
    # assert by membership rather than order to keep the test resilient
    # to the threshold layer's tie-break shape — and assert order
    # separately on the structural-merge test below where the marker
    # counts produce an unambiguous ranking.
    assert len(hunks) == 3
    by_kind = {hunk["kind"]: hunk for hunk in hunks}
    assert set(by_kind) == {
        "calibration_non_fit_pattern",
        "calibration_depth_distinction",
        "calibration_examples",
    }

    non_fit = by_kind["calibration_non_fit_pattern"]
    assert non_fit["section"] == "non_fit_patterns"
    assert non_fit["target_field"] == "non_fit_patterns"
    assert "Foundation Models Research" in non_fit["label"]
    assert "Foundation Models Research" in non_fit["after"]
    assert non_fit["default_approved"] is False
    assert non_fit["confidence"] < 0.65  # NEEDS-REVIEW at Gate 2

    depth = by_kind["calibration_depth_distinction"]
    assert depth["section"] == "depth_distinction"
    assert depth["target_field"] == "depth_distinction"
    assert "Inference Optimization" in depth["label"]
    assert "depth_distinction.edge_case_guidance" in depth["after"]
    assert "Inference Optimization" in depth["after"]
    assert depth["default_approved"] is False

    cal = by_kind["calibration_examples"]
    assert cal["section"] == "transferability_examples"
    assert cal["target_field"] == "transferability_examples"
    assert "Frontend Performance" in cal["label"]
    assert "transfers" in cal["after"]
    assert cal["default_approved"] is False

    # Wire-shape contract — every hunk must carry the keys the frontend's
    # ``normalizeHunk`` (cloris/frontend/src/lib/reflection/types.ts)
    # expects to be present and well-typed. Drift here would silently
    # break Gate-2 rendering for calibration hunks.
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
        assert hunk["hunk_id"].startswith("calibration-")
        assert isinstance(hunk["confidence"], float)
        assert isinstance(hunk["default_approved"], bool)


def test_helper_surfaces_existing_value_in_before_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the brief already carries a non_fit_patterns entry for the
    same capability area, the calibration hunk surfaces it in
    ``before`` so the recruiter sees what would change at Gate 2.

    Pins the diff-rendering contract: an "add" calibration hunk that
    happens to overlap with an existing brief entry is still a
    proposal (the translator decides shape; the apply layer decides
    merge), but the recruiter must SEE the existing value to reason
    about whether the proposal duplicates or refines.
    """

    state_dir = tmp_path / "linkedin" / "state-key-existing"
    store = _make_store(state_dir)
    brief_id = "brief-with-existing-non-fit"
    for identity_key in ("a", "b", "c", "d", "e"):
        _seed_marker(
            store,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            capability_area="ML Tooling",
            confidence=0.6,
            terminal_decision="REJECT",
            judgment_accuracy="wrong",
        )

    _stub_state_dirs_for_brief_id(monkeypatch, [("linkedin", state_dir)])

    brief_raw = {
        "id": brief_id,
        "non_fit_patterns": [
            {
                "label": "ML Tooling",
                "description": "Existing entry from prior reflection cycle.",
                "why_not": "Recruiter previously said this surfaces noise.",
                "examples": [],
            }
        ],
    }

    hunks = _calibration_propose_hunks(
        brief=_StubBrief(brief_id=brief_id),
        brief_raw=brief_raw,
        brief_path=tmp_path / "brief.json",
    )

    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk["kind"] == "calibration_non_fit_pattern"
    # The existing entry's label + description surface in ``before``.
    assert "ML Tooling" in hunk["before"]
    assert "Existing entry from prior reflection cycle." in hunk["before"]
    # The proposed update lands in ``after``.
    assert "ML Tooling" in hunk["after"]


# ---------------------------------------------------------------------------
# Defensive: helper never crashes the reflection phase
# ---------------------------------------------------------------------------


def test_helper_returns_empty_when_brief_has_no_id() -> None:
    """A brief without an ``id`` (e.g., scratch / draft) cannot be
    matched against runtime-state ``brief_id``; helper returns ``[]``
    rather than crashing.
    """

    hunks = _calibration_propose_hunks(
        brief=_StubBrief(brief_id=""),
        brief_raw={},
        brief_path=Path("/nonexistent/brief.json"),
    )
    assert hunks == []


def test_helper_returns_empty_when_no_state_dirs_match_brief_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brief whose runs haven't started yet has no matching state
    dirs; helper returns ``[]`` (no markers → no patches).
    """

    _stub_state_dirs_for_brief_id(monkeypatch, [])

    hunks = _calibration_propose_hunks(
        brief=_StubBrief(brief_id="brief-no-runs"),
        brief_raw={"id": "brief-no-runs"},
        brief_path=tmp_path / "brief.json",
    )
    assert hunks == []


def test_helper_returns_empty_when_state_dir_has_no_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State dir exists but no candidate has ``judgment_accuracy``
    set yet (recruiter hasn't marked anything). Helper returns ``[]``.
    """

    state_dir = tmp_path / "linkedin" / "state-empty"
    state_dir.mkdir(parents=True)
    # Touch an empty SQLite file so the aggregator's ``mode=ro`` open
    # succeeds and the SELECT returns 0 rows (vs. the missing-DB
    # branch which is its own coverage at test_calibration_aggregator).
    _make_store(state_dir)

    _stub_state_dirs_for_brief_id(monkeypatch, [("linkedin", state_dir)])

    hunks = _calibration_propose_hunks(
        brief=_StubBrief(brief_id="brief-zero"),
        brief_raw={"id": "brief-zero"},
        brief_path=tmp_path / "brief.json",
    )
    assert hunks == []


def test_helper_returns_empty_when_no_area_clears_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An area with markers but below the per-area weighted floor
    (5 weighted markers) doesn't surface a hunk.

    Two `useful` markers in one area = 2 weighted (no bonus on
    `useful`); below threshold → empty.
    """

    state_dir = tmp_path / "linkedin" / "state-thin"
    store = _make_store(state_dir)
    brief_id = "brief-too-thin"
    for identity_key in ("a", "b"):
        _seed_marker(
            store,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            capability_area="Quiet Area",
            confidence=0.7,
            terminal_decision="SAVE",
            judgment_accuracy="useful",
        )

    _stub_state_dirs_for_brief_id(monkeypatch, [("linkedin", state_dir)])

    hunks = _calibration_propose_hunks(
        brief=_StubBrief(brief_id=brief_id),
        brief_raw={"id": brief_id},
        brief_path=tmp_path / "brief.json",
    )
    assert hunks == []


# ---------------------------------------------------------------------------
# Cross-source merge
# ---------------------------------------------------------------------------


def test_helper_merges_rollups_across_state_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brief that lives in two state-dirs (e.g., LinkedIn + GitHub)
    has its rollups merged before threshold + translator run, so the
    per-cycle cap is enforced once across sources rather than twice.

    Seeds 3 ``wrong`` markers in ``Distributed Systems`` on the LinkedIn
    state-dir + 3 more on the GitHub state-dir. Per-source neither
    crosses the per-area floor (5 weighted), but the merged view does
    (6 weighted). One non_fit_pattern hunk surfaces.
    """

    li_state_dir = tmp_path / "linkedin" / "state-1"
    gh_state_dir = tmp_path / "github" / "state-2"
    li_store = _make_store(li_state_dir)
    gh_store = _make_store(gh_state_dir)

    brief_id = "brief-multi-source"
    for identity_key in ("li-1", "li-2", "li-3"):
        _seed_marker(
            li_store,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            capability_area="Distributed Systems",
            confidence=0.6,
            terminal_decision="REJECT",
            judgment_accuracy="wrong",
        )
    for identity_key in ("gh-1", "gh-2", "gh-3"):
        _seed_marker(
            gh_store,
            source="github",
            brief_id=brief_id,
            identity_key=identity_key,
            capability_area="Distributed Systems",
            confidence=0.6,
            terminal_decision="REJECT",
            judgment_accuracy="wrong",
        )

    _stub_state_dirs_for_brief_id(
        monkeypatch,
        [("linkedin", li_state_dir), ("github", gh_state_dir)],
    )

    hunks = _calibration_propose_hunks(
        brief=_StubBrief(brief_id=brief_id),
        brief_raw={"id": brief_id},
        brief_path=tmp_path / "brief.json",
    )

    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk["kind"] == "calibration_non_fit_pattern"
    assert "Distributed Systems" in hunk["after"]
    # Marker count visible in the rendered text proves both sources
    # contributed (3 + 3 = 6, not 3).
    assert "6" in hunk["after"]


# ---------------------------------------------------------------------------
# Slice's actual ask: calibration hunks surface ALONGSIDE
# brief-recommendation hunks
# ---------------------------------------------------------------------------


def test_calibration_hunks_merge_alongside_brief_recommendation_hunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The slice card's load-bearing assertion (lines 973-975):
    ``reflection_phase_propose surfaces calibration hunks alongside
    brief-recommendation hunks``.

    Mirrors the propose phase's exact merge sequence at
    ``market_intelligence/reflection.py:reflection_phase_propose``:

    1. ``hunks = _build_hunks_from_artifact(artifact_dict, brief_raw=raw)``
    2. ``hunks.extend(_calibration_propose_hunks(...))``
    3. ``hunks.extend(_designer_rubric_refine_propose_hunks(...))``

    Verifies the merged list carries both kinds with stable identifiable
    shapes (kind strings + hunk_id prefixes don't collide) and that
    the brief-recommendation hunks come first. Order matters for Gate-2
    UX — recruiters scan top-down; calibration patches are derived
    signal layered on top of the artifact's primary brief
    recommendations.
    """

    # Set up calibration markers for one capability area.
    state_dir = tmp_path / "linkedin" / "state-merged"
    store = _make_store(state_dir)
    brief_id = "brief-merged-test"
    for identity_key in ("m1", "m2", "m3", "m4", "m5"):
        _seed_marker(
            store,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            capability_area="Causal Inference",
            confidence=0.6,
            terminal_decision="REJECT",
            judgment_accuracy="wrong",
        )
    _stub_state_dirs_for_brief_id(monkeypatch, [("linkedin", state_dir)])

    # Build a synthetic artifact carrying brief-recommendation entries
    # the way the propose phase's _build_artifact would.
    artifact_dict = {
        "brief_recommendations": [
            {
                "recommendation_id": "rec-1",
                "target_field": "additional_search_terms",
                "proposal": "MLOps platform",
                "reason": "Adjacent talent pool worth scanning",
                "confidence": 0.8,
            },
        ]
    }
    brief_raw: dict = {"id": brief_id, "additional_search_terms": []}

    # Mirror reflection_phase_propose's exact merge sequence.
    hunks = _build_hunks_from_artifact(artifact_dict, brief_raw=brief_raw)
    hunks.extend(
        _calibration_propose_hunks(
            brief=_StubBrief(brief_id=brief_id),
            brief_raw=brief_raw,
            brief_path=tmp_path / "brief.json",
        )
    )

    assert len(hunks) == 2
    rec_hunk, calibration_hunk = hunks

    # Brief-recommendation hunk first — comes from the artifact,
    # which is the propose phase's primary surface.
    assert rec_hunk["section"] == "additional_search_terms"
    assert rec_hunk["after"] == "MLOps platform"
    assert rec_hunk["hunk_id"] == "rec-1"
    assert rec_hunk["default_approved"] is True  # 0.8 >= 0.65

    # Calibration hunk second — derived signal, NEEDS-REVIEW by default.
    assert calibration_hunk["kind"] == "calibration_non_fit_pattern"
    assert calibration_hunk["hunk_id"] == "calibration-non_fit_pattern-1"
    assert calibration_hunk["section"] == "non_fit_patterns"
    assert calibration_hunk["default_approved"] is False
    assert calibration_hunk["confidence"] < 0.65

    # The two hunk-id namespaces don't collide. ``rec-*`` is the
    # brief-recommendation namespace (driven by ``recommendation_id``);
    # ``calibration-*`` is the calibration namespace. Different
    # namespaces protect against frontend per-hunk-state collisions
    # at Gate 2.
    assert rec_hunk["hunk_id"] != calibration_hunk["hunk_id"]
    assert not rec_hunk["hunk_id"].startswith("calibration-")
    assert not calibration_hunk["hunk_id"].startswith("rec-")
