"""A candidate released for re-judging can actually re-enter the pipeline.

Two subsystems disagreed, and the disagreement killed live sessions.

`_HASH_AWARE_SUPPRESSION_CLAUSE` deliberately RELEASES a non-SAVE terminal
verdict that was made under a different `brief_content_hash`, so a revised brief
can re-judge the person. But `ALLOWED_LIFECYCLE_TRANSITIONS` had
`full_terminal: set()` and no `snippet_extracted` exit from `facial_terminal`,
so the first re-encounter raised:

    ValueError: invalid lifecycle transition: full_terminal -> snippet_extracted

which is not caught anywhere on the sourcing path — it unwinds through
`_review_page_batch` → `_process_string_impl` → `run_full` and ends the session.
Measured 2026-07-27: a live run died six minutes in on string 1, page 1, card 2,
and a second launch was blocked by 12 more released rows.

P3.1 had already established the principle for `failed_terminal` ("re-eligible
after a brief revision may re-enter the pipeline"); the judgment terminals were
simply left behind.
"""

from __future__ import annotations

import pytest

from shared.runtime_state.store import (
    ALLOWED_LIFECYCLE_TRANSITIONS,
    DEDUP_BLOCKING_DECISIONS,
    SAVE_FAMILY_DECISIONS,
    RuntimeStateStore,
    _guard_transition,
)


@pytest.fixture()
def store(tmp_path):
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


@pytest.mark.parametrize("terminal", ["full_terminal", "facial_terminal", "failed_terminal"])
def test_every_terminal_state_can_re_enter_at_snippet(terminal: str) -> None:
    # The whole bug in one assertion. Each of these is reachable by a candidate
    # the suppression clause has released, and each must be able to start over.
    _guard_transition(terminal, "snippet_extracted")


def test_no_terminal_state_is_absorbing() -> None:
    for state in ("full_terminal", "facial_terminal", "failed_terminal"):
        assert ALLOWED_LIFECYCLE_TRANSITIONS[state], f"{state} is a dead end"


def test_the_release_rule_and_the_state_machine_agree(store) -> None:
    """End to end through the real store: judge, revise the brief, re-encounter.

    This is the sequence that killed the live run.
    """
    run_id = store.start_run(
        source="linkedin", brief_id="b1", output_dir="/tmp/x", mode="fresh",
    )
    work_unit_id = store.upsert_work_unit(
        run_id=run_id, source="linkedin", brief_id="b1", kind="linkedin_string",
        source_unit_id="1", display_name="string 1", ordering_index=0,
        status="in_progress",
    )

    # Judged and rejected under the ORIGINAL brief.
    attempt = store.start_attempt(
        run_id=run_id, source="linkedin", brief_id="b1",
        identity_key="in/someone", stage="snippet", work_unit_id=work_unit_id,
        display_name="Someone", profile_url="https://linkedin.com/in/someone",
    )
    store.finish_attempt_success(attempt_id=attempt, new_state="snippet_extracted", run_id=run_id)
    # The real forward path, one legal step at a time.
    for stage, state in (
        ("facial", "facial_started"),
        ("facial", "facial_terminal"),
        ("full", "full_started"),
    ):
        a = store.start_attempt(
            run_id=run_id, source="linkedin", brief_id="b1",
            identity_key="in/someone", stage=stage, work_unit_id=work_unit_id,
        )
        store.finish_attempt_success(attempt_id=a, new_state=state, run_id=run_id)
    a = store.start_attempt(
        run_id=run_id, source="linkedin", brief_id="b1",
        identity_key="in/someone", stage="full", work_unit_id=work_unit_id,
    )
    store.finish_attempt_success(
        attempt_id=a, new_state="full_terminal", terminal_decision="REJECT", run_id=run_id,
    )

    # The brief is revised, so the old REJECT no longer governs — this is the
    # release the suppression clause is designed to perform.
    reentry = store.start_attempt(
        run_id=run_id, source="linkedin", brief_id="b1",
        identity_key="in/someone", stage="snippet", work_unit_id=work_unit_id,
    )
    store.finish_attempt_success(
        attempt_id=reentry, new_state="snippet_extracted", run_id=run_id,
    )  # raised ValueError before the fix, killing the whole session


def test_a_saved_candidate_is_never_released_for_re_judging() -> None:
    # The safety argument the transition change rests on. A SAVE-family verdict
    # suppresses regardless of brief hash, so a saved person can never reach a
    # re-snippet and never be re-saved. If this ever stops holding, the
    # transition above becomes a double-save risk.
    for decision in SAVE_FAMILY_DECISIONS:
        assert decision in DEDUP_BLOCKING_DECISIONS


def test_the_forward_path_is_still_guarded() -> None:
    # Re-entry must not become "anything goes". A judgment may not skip
    # straight to a verdict without the intervening started state.
    # (2026-07-28: `full_terminal -> full_started` moved OUT of this list —
    # stage-complete re-entry made it legal after run 7 died on the sibling
    # edge `full_terminal -> facial_started`. Forward SKIPS remain the crime;
    # re-entering a started stage is not a skip.)
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        _guard_transition("snippet_extracted", "full_terminal")
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        _guard_transition("discovered", "facial_terminal")


@pytest.mark.parametrize("terminal", ["full_terminal", "facial_terminal", "failed_terminal"])
@pytest.mark.parametrize("reentry", ["snippet_extracted", "facial_started", "full_started"])
def test_reentry_is_stage_complete(terminal: str, reentry: str) -> None:
    # 2026-07-28, run 7: the 07-27 fix opened only snippet_extracted, and the
    # very next session died on `full_terminal -> facial_started` when batch
    # facial dispatched against a re-encountered terminal candidate. A released
    # candidate re-enters wherever the pipeline actually meets them, so every
    # terminal state must reach every started stage.
    if terminal == "failed_terminal" and reentry == "full_started":
        pass  # already allowed pre-P3.1-completion; covered for symmetry
    _guard_transition(terminal, reentry)


def test_forward_skips_are_still_illegal_after_stage_complete_reentry() -> None:
    for current, target in (
        ("discovered", "facial_terminal"),
        ("discovered", "full_terminal"),
        ("snippet_extracted", "facial_terminal"),
        ("snippet_extracted", "full_terminal"),
        ("facial_started", "full_terminal"),
    ):
        with pytest.raises(ValueError, match="invalid lifecycle transition"):
            _guard_transition(current, target)


def test_review_decisions_now_suppress_under_the_same_brief(store) -> None:
    # The other half of run 7's crash: REVIEW_FLAGGED/REVIEW_INFERRED were the
    # one terminal class outside DEDUP_BLOCKING_DECISIONS, so a review-parked
    # candidate was re-processed by every string that surfaced them — Jingran
    # Zhou drew a second facial 23 minutes after being review-parked. Same
    # brief: suppressed. Changed brief: the hash-aware clause releases them,
    # exactly like any other non-SAVE verdict.
    assert "REVIEW_FLAGGED" in DEDUP_BLOCKING_DECISIONS
    assert "REVIEW_INFERRED" in DEDUP_BLOCKING_DECISIONS
    # And they are NOT save-family — a brief change must still release them.
    assert "REVIEW_FLAGGED" not in SAVE_FAMILY_DECISIONS
    assert "REVIEW_INFERRED" not in SAVE_FAMILY_DECISIONS


def test_save_family_terminals_refuse_reentry_released_rejects_do_not(store) -> None:
    # The decision-aware belt that replaced the state-only one: stage-complete
    # re-entry exists for RELEASED candidates, and release never applies to
    # SAVE-family verdicts — so a save reaching a re-entry write means
    # suppression failed upstream, and the store refuses rather than letting
    # the terminal_decision be overwritten (the double-save door).
    from shared.runtime_state.store import _guard_save_family_reentry

    for stage in ("snippet_extracted", "facial_started", "full_started"):
        with pytest.raises(ValueError, match="never *\n?.*re-entered|never "):
            _guard_save_family_reentry("full_terminal", "SAVE", stage)
        # A released REJECT re-enters freely at the same stages.
        _guard_save_family_reentry("full_terminal", "REJECT", stage)
        _guard_save_family_reentry("facial_terminal", "FACIAL_NO", stage)
