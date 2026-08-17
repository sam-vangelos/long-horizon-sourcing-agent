"""P6.8 — github reflection wiring into the propose phase.

Before this fix, ``market_intelligence.github_reflection.propose_github_hunks``
had no production caller anywhere in the codebase — it was a fully unit-
tested composer (see ``tests/test_per_module_reflection_hunks.py``) that
nothing in ``reflection_phase_propose`` ever invoked. This file pins:

1. ``market_intelligence.reflection._github_reflection_propose_hunks`` —
   the new wiring helper that pools ``final_judgments`` across every
   github-source :class:`~market_intelligence.schema.MarketEvidenceBatch`
   for a run and hands them to ``propose_github_hunks``.
2. A source-contract regression lock on ``reflection_phase_propose``
   itself: the designer rubric-refinement composer's *call* has been
   replaced by the github composer's call in that function body (the
   designer composer function itself is untouched and still directly
   unit-tested by ``tests/test_designer_rubric_refinement_wiring.py``).
"""

from __future__ import annotations

import inspect

from market_intelligence.reflection import (
    _designer_rubric_refine_propose_hunks,
    _github_reflection_propose_hunks,
    reflection_phase_propose,
)
from market_intelligence.schema import MarketEvidenceBatch


def _batch(
    *,
    source: str,
    run_ref: str,
    output_dir: str = "/tmp/fixture",
    final_judgments: list[dict] | None = None,
) -> MarketEvidenceBatch:
    return MarketEvidenceBatch(
        run_ref=run_ref,
        source=source,
        output_dir=output_dir,
        brief_version="2.0",
        generated_at="2026-05-03T00:00:00Z",
        final_judgments=final_judgments or [],
    )


def _maintainer_save(*, project: str, level: str, confidence: float = 0.8) -> dict:
    """Mirrors the shape ``github_reflection._iter_save_records`` reads
    (same fixture convention as ``tests/test_per_module_reflection_hunks.py``)."""

    return {
        "decision": "SAVE",
        "maintainership": {
            "level": level,
            "confidence": confidence,
            "signals": {"commit_count": 50, "merge_authority": 1},
            "evidence_sources": [
                f"commit_count:{project}",
                f"merge_authority:{project}",
            ],
        },
    }


# ---------------------------------------------------------------------------
# _github_reflection_propose_hunks — the wiring helper
# ---------------------------------------------------------------------------


def test_returns_empty_when_no_github_evidence_batch() -> None:
    """A classic LinkedIn-only run has no github batch — no-op, not a crash."""

    linkedin_batch = _batch(
        source="linkedin",
        run_ref="linkedin:run1",
        final_judgments=[_maintainer_save(project="kubernetes/kubernetes", level="maintainer")],
    )
    out = _github_reflection_propose_hunks(
        brief_raw={}, evidence_batches=[linkedin_batch]
    )
    assert out == []


def test_returns_empty_when_github_batch_has_no_final_judgments() -> None:
    gh_batch = _batch(source="github", run_ref="github:run1", final_judgments=[])
    out = _github_reflection_propose_hunks(brief_raw={}, evidence_batches=[gh_batch])
    assert out == []


def test_returns_empty_when_evidence_batches_list_is_empty() -> None:
    assert _github_reflection_propose_hunks(brief_raw={}, evidence_batches=[]) == []


def test_pools_final_judgments_across_multiple_github_batches() -> None:
    """A brief can have multiple github run dirs (current run + imported
    legacy runs per ``_collect_evidence_batches``) — the wiring helper
    must pool across all of them, not just the first."""

    gh1 = _batch(
        source="github",
        run_ref="github:run1",
        final_judgments=[
            _maintainer_save(project="rust-lang/rust", level="maintainer")
            for _ in range(6)
        ],
    )
    gh2 = _batch(
        source="github",
        run_ref="github:run2",
        final_judgments=[
            _maintainer_save(project="kubernetes/kubernetes", level="maintainer")
            for _ in range(3)
        ],
    )
    brief_raw = {"target_projects": ["kubernetes/kubernetes"]}

    hunks = _github_reflection_propose_hunks(
        brief_raw=brief_raw, evidence_batches=[gh1, gh2]
    )

    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk["section"] == "target_projects"
    # rust-lang/rust only shows up because run2's kubernetes saves were
    # pooled in too (9 total saves; 6/9 clears the notable-cluster bar) —
    # a single-batch implementation would only see 6 saves from gh1 and
    # would still fire, so also assert the pooled *count* landed in the
    # rationale to prove both batches were actually combined.
    assert "9 total saves" in hunk["rationale"]


def test_hunks_are_needs_review_by_default_never_auto_write() -> None:
    """Doctrine: hunks are Gate-2 proposals only. Both github hunk kinds
    ship at confidence below the 0.65 default_approved threshold."""

    gh = _batch(
        source="github",
        run_ref="github:run1",
        final_judgments=(
            [
                _maintainer_save(project="kubernetes/kubernetes", level="maintainer")
                for _ in range(5)
            ]
            + [
                _maintainer_save(project="kubernetes/kubernetes", level="contributor")
                for _ in range(3)
            ]
        ),
    )
    brief_raw = {
        "target_projects": ["kubernetes/kubernetes"],
        "maintainership_level": "project_lead",
    }

    hunks = _github_reflection_propose_hunks(brief_raw=brief_raw, evidence_batches=[gh])

    assert hunks, "expected the lower-maintainership-threshold hunk to fire"
    for hunk in hunks:
        assert hunk["default_approved"] is False
        assert hunk["confidence"] < 0.65


# ---------------------------------------------------------------------------
# reflection_phase_propose — source-contract regression lock
# ---------------------------------------------------------------------------
#
# A full reflection_phase_propose() call requires a live artifact-build +
# synthesis/critic-backend + planner-result pipeline (see the heavy setup
# tests_reflection_calibration_integration.py deliberately avoids for its
# own "hunks merge alongside" assertion, replicating the merge sequence
# manually instead). We do the same here: assert the exact wiring change
# at the source level, which is the load-bearing fact P6.8 is about (the
# composer existed but had zero callers).


def test_propose_phase_calls_github_composer_not_designer_composer() -> None:
    """P6.8 regression lock: ``reflection_phase_propose`` must call
    ``_github_reflection_propose_hunks`` and must NOT call
    ``_designer_rubric_refine_propose_hunks`` (designer is sunset; its
    composer function is left intact and separately tested, but no
    longer wired into the propose phase).
    """

    source = inspect.getsource(reflection_phase_propose)
    assert "_github_reflection_propose_hunks(" in source
    assert "_designer_rubric_refine_propose_hunks(" not in source


def test_designer_composer_function_still_exists_and_is_callable() -> None:
    """The spec explicitly forbids deleting the designer composer —
    only its call site changes. Confirm it's still there, still
    importable, and still behaves per its own contract (empty brief ⇒
    empty hunks) — i.e. genuinely preserved, not left as dead code that
    happens to still parse."""

    out = _designer_rubric_refine_propose_hunks(
        brief_raw={"target_modules": ["linkedin"]}, brief_path=None
    )
    assert out == []
