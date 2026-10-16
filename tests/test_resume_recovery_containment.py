"""Resume pending-full recovery containment (flag-gated).

LinkedIn reorders Recruiter results between sessions, so a resume can fail to
relocate the exact card a facial YES/BORDERLINE was captured against. The raise
that followed was deterministic across retries and killed every resume attempt
of the 2026-07-31 campaign. These tests lock both flag states: OFF still raises,
ON terminally settles the candidate on canonical state, receipts the skip, and
lets the remaining owned snippets recover.

Run with: python -m pytest tests/test_resume_recovery_containment.py -v
"""

import asyncio
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared import config
from shared.runtime_state.read_models import has_pending_work
from shared.schemas import OpusDecision, Progress, SearchString

from tests.test_linkedin_pipeline import _make_pipeline, _make_snippet


_ABANDON_EVENT = "pending_full_recovery_abandoned"


def _seed_facial_yes(p, search_string: SearchString, snippet, decision="FACIAL_YES"):
    """Persist the succeeded facial attempt that makes a candidate pending.

    Canonical rehydration only reads succeeded attempts, and the runtime store
    refuses a stage write against a candidate it has never seen — so this is
    also the registration a live pending candidate always carries by the time
    recovery meets it.
    """

    p.state.reviewer.attempts._record_runtime_snippet(search_string, snippet)
    facial_attempt_id = p.state.reviewer.attempts._start_runtime_stage_attempt(
        search_string=search_string,
        snippet=snippet,
        stage="facial",
    )
    p.state.reviewer._finish_runtime_stage_success(
        attempt_id=facial_attempt_id,
        stage="facial",
        snippet=snippet,
        decision=OpusDecision(
            stage="facial",
            decision=decision,
            path="normal_eligibility",
            confidence=0.9,
            rationale="pending full review",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        ),
    )


def _track_pending(p, snippet, owner: SearchString, decision="FACIAL_YES") -> str:
    assert snippet.source_string_id == owner.id
    key = p.state.candidates._funnel_candidate_key(snippet)
    p.state.candidates._track_full_review_obligation(snippet, decision)
    return key


def _settle_full(p, search_string: SearchString, snippet, decision: str) -> None:
    """Persist a real succeeded full verdict — the re-meet a later run produces."""

    full_attempt_id = p.state.reviewer.attempts._start_runtime_stage_attempt(
        search_string=search_string,
        snippet=snippet,
        stage="full",
    )
    verdict = OpusDecision(
        stage="full",
        decision=decision,
        path="DIRECT:re-met",
        confidence=0.8,
        rationale="re-met on a later surface and actually evaluated",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
    )
    p.state.reviewer._finish_runtime_stage_success(
        attempt_id=full_attempt_id,
        stage="full",
        snippet=snippet,
        decision=verdict,
    )


def _close_all_work_units(p) -> None:
    """Retire the string work units so has_pending_work reaches the review CTE."""

    with p.state.context._runtime_state.connect() as conn:
        conn.execute("UPDATE work_units SET status = 'done'")


def _abandon_events(p) -> list[dict]:
    with p.state.context._runtime_state.connect() as conn:
        rows = conn.execute(
            "SELECT payload_json FROM events WHERE event_type = ? ORDER BY id",
            (_ABANDON_EVENT,),
        ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def test_recover_pending_full_unmatched_raises_when_containment_disabled(monkeypatch):
    """Flag off still kills resume on an unmatched card.

    The flag defaults ON since 2026-09-20, so this proof forces it off rather
    than leaning on the default — the kill switch keeps its coverage.
    """

    monkeypatch.setattr(config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", False)

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=3,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        pending = _make_snippet(
            name="Unmatched Pending",
            profile_url="/talent/profile/unmatched-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        key = _track_pending(p, pending, owner)
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()

        with pytest.raises(
            RuntimeError,
            match="could not match the exact Recruiter profile",
        ):
            asyncio.run(
                p.state.driver._recover_owner_pending_full_evaluations(
                    progress=progress,
                    search_string=owner,
                    first_incomplete_page=2,
                    string_stats=p.state.candidates._fresh_string_stats(),
                )
            )

        assert owner.status == "in_progress"
        p.state._allocator_service._checkpoint_progress.assert_called_once_with(progress)
        p.state.reviewer._process_resumed_pending_full_evaluations.assert_not_awaited()
        assert key in p.state.candidates._resume_pending_full_decisions
        assert key in p.state.candidates._resume_pending_full_snippets
        assert key in p.state.candidates._resume_pending_full_owner_ids


def test_recover_pending_full_unmatched_contained_skips_and_receipts(
    monkeypatch,
    capsys,
):
    """Flag on: the unmatched candidate is receipted and the rest still recover."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=4,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        unmatched = _make_snippet(
            name="Unmatched Pending",
            profile_url="/talent/profile/unmatched-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
            result_rank=1,
        )
        survivor = _make_snippet(
            name="Matched Pending",
            profile_url="/talent/profile/matched-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
            result_rank=2,
        )
        _seed_facial_yes(p, owner, unmatched)
        _seed_facial_yes(p, owner, survivor, decision="FACIAL_BORDERLINE")
        unmatched_key = _track_pending(p, unmatched, owner)
        survivor_key = _track_pending(
            p,
            survivor,
            owner,
            decision="FACIAL_BORDERLINE",
        )

        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(
            side_effect=[None, 3]
        )
        p.state._allocator_service._checkpoint_progress = MagicMock()
        recovered: list[str] = []

        async def settle_pending(**kwargs):
            snippet = kwargs["snippets"][0]
            recovered.append(snippet.profile_url)
            p.state.candidates.forget_pending(snippet.profile_url)
            return False

        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock(
            side_effect=settle_pending
        )

        asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=2,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )

        # (e) the loop proceeds: the matched candidate still recovers.
        assert recovered == [survivor.profile_url]
        assert survivor_key not in p.state.candidates._resume_pending_full_decisions
        # (c) all three maps release the contained candidate.
        assert unmatched_key not in p.state.candidates._resume_pending_full_decisions
        assert unmatched_key not in p.state.candidates._resume_pending_full_snippets
        assert unmatched_key not in p.state.candidates._resume_pending_full_owner_ids
        # (b) the receipt carries every payload field an operator re-runs from.
        payloads = _abandon_events(p)
        assert payloads == [
            {
                "string_id": owner.id,
                "page": 1,
                "candidate_name": unmatched.name,
                "profile_url": unmatched.profile_url,
                "reason": "unmatched_profile",
                "status": "failed",
            }
        ]
        # (d) one console line in the existing [recover-full] style.
        out = capsys.readouterr().out
        assert (
            f"[recover-full] SKIPPED {unmatched.name} — unmatched_profile; "
            f"exact string #{owner.id} page 1" in out
        )


def test_recover_pending_full_missing_url_contained_skips_and_receipts(
    monkeypatch,
    capsys,
):
    """Same contract for the URL-less pending row, reason ``missing_profile_url``.

    ``_validated_owner_pending_full_snippets`` already fails closed on an empty
    profile URL, so the in-loop branch is defensive depth. Stubbing the
    validator is the only way to reach it.
    """

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=5,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        urlless = _make_snippet(
            name="Urlless Pending",
            profile_url="",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        key = _track_pending(p, urlless, owner)
        p.state.driver._validated_owner_pending_full_snippets = MagicMock(
            return_value=[(key, urlless)]
        )
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=7)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()

        asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=2,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )

        p.state.context.browser.find_result_slot_by_profile_url.assert_not_awaited()
        p.state.reviewer._process_resumed_pending_full_evaluations.assert_not_awaited()
        assert key not in p.state.candidates._resume_pending_full_decisions
        assert key not in p.state.candidates._resume_pending_full_snippets
        assert key not in p.state.candidates._resume_pending_full_owner_ids
        assert _abandon_events(p) == [
            {
                "string_id": owner.id,
                "page": 1,
                "candidate_name": urlless.name,
                "profile_url": "",
                "reason": "missing_profile_url",
                "status": "failed",
            }
        ]
        out = capsys.readouterr().out
        assert (
            f"[recover-full] SKIPPED {urlless.name} — missing_profile_url; "
            f"exact string #{owner.id} page 1" in out
        )


def test_recover_pending_full_contained_settlement_survives_rehydration(
    monkeypatch,
):
    """The durable proof: the REAL rehydration no longer lists the candidate.

    Runs ``_hydrate_resume_funnel_from_runtime`` — the derivation a resuming
    process actually uses — before and after containment. Nothing about the
    pending derivation is mocked.
    """

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=6,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        pending = _make_snippet(
            name="Unmatched Pending",
            profile_url="/talent/profile/unmatched-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        _seed_facial_yes(p, owner, pending)

        p.state.candidates._hydrate_resume_funnel_from_runtime(progress, store=p.state.context._runtime_state, run_id=p.state.context._runtime_run_id, brief_id=p.state.context._brief_id, stats=p.state.context.stats)
        key = p.state.candidates._funnel_candidate_key(pending)
        assert key in p.state.candidates._resume_pending_full_decisions
        assert p.state.candidates._resume_pending_full_owner_ids[key] == owner.id
        assert owner.full_reviewed_count == 0

        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()

        asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=2,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )
        p.state.reviewer._process_resumed_pending_full_evaluations.assert_not_awaited()

        with p.state.context._runtime_state.connect() as conn:
            settled = conn.execute(
                "SELECT ca.status, ca.payload_json, c.current_lifecycle_state "
                "FROM candidate_attempts ca "
                "JOIN candidates c ON c.id = ca.candidate_id "
                "WHERE ca.stage = 'full' AND c.identity_key = ?",
                (pending.profile_url,),
            ).fetchall()
        assert len(settled) == 1
        assert settled[0]["status"] == "succeeded"
        assert settled[0]["current_lifecycle_state"] == "full_terminal"
        settled_payload = json.loads(settled[0]["payload_json"])
        assert settled_payload["full_decision"]["decision"] == "JUDGMENT_FAILURE"
        assert settled_payload["abandon_reason"] == "unmatched_profile"
        assert settled_payload["pending_full_recovery_abandoned"] is True

        # The derivation itself, re-run exactly as a resuming process runs it.
        p.state.candidates._hydrate_resume_funnel_from_runtime(progress, store=p.state.context._runtime_state, run_id=p.state.context._runtime_run_id, brief_id=p.state.context._brief_id, stats=p.state.context.stats)
        assert p.state.candidates._resume_pending_full_decisions == {}
        assert p.state.candidates._resume_pending_full_snippets == {}
        assert p.state.candidates._resume_pending_full_owner_ids == {}
        # A skip is not a review: the abandoned row stays out of the counters.
        assert owner.facial_yes_count == 1
        assert owner.full_reviewed_count == 0
        assert owner.full_reject_count == 0
        assert p.state.context.stats["full_reviewed"] == 0


def test_recover_pending_full_pagination_exhaustion_still_raises_with_flag_on(
    monkeypatch,
):
    """Containment is scoped to the two per-candidate re-match failures only."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=8,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=3,
        )
        progress = Progress(brief_name="test", strings=[owner])
        pending = _make_snippet(
            name="Page Two Pending",
            profile_url="/talent/profile/page-two-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=2,
        )
        key = _track_pending(p, pending, owner)
        p.state.surface._go_to_next_page_with_transient_retry = AsyncMock(
            return_value=(False, False)
        )
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()

        with pytest.raises(
            RuntimeError,
            match="pagination exhaustion before pending full review page 2",
        ):
            asyncio.run(
                p.state.driver._recover_owner_pending_full_evaluations(
                    progress=progress,
                    search_string=owner,
                    first_incomplete_page=3,
                    string_stats=p.state.candidates._fresh_string_stats(),
                )
            )

        assert owner.status == "in_progress"
        p.state._allocator_service._checkpoint_progress.assert_called_once_with(progress)
        p.state.context.browser.find_result_slot_by_profile_url.assert_not_awaited()
        assert key in p.state.candidates._resume_pending_full_decisions


def _contain_one(p, owner: SearchString, progress: Progress) -> None:
    """Run recovery with the live surface unable to re-match the pending card."""

    p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)
    p.state._allocator_service._checkpoint_progress = MagicMock()
    p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()
    asyncio.run(
        p.state.driver._recover_owner_pending_full_evaluations(
            progress=progress,
            search_string=owner,
            first_incomplete_page=2,
            string_stats=p.state.candidates._fresh_string_stats(),
        )
    )
    p.state.reviewer._process_resumed_pending_full_evaluations.assert_not_awaited()


def test_contained_row_yields_to_a_later_real_full_decision(monkeypatch):
    """The abandon placeholder must never shadow a verdict a judge actually made.

    Under plain first-succeeded-wins the synthetic JUDGMENT_FAILURE would own
    the candidate forever, so a re-meet that really evaluated the person would
    vanish from the counters. Control: the same facial + REJECT with no
    containment in between must produce identical numbers.
    """

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as ctrl_td:
        p = _make_pipeline(td)
        owner = SearchString(id=11, name="owner", boolean="ml", status="done")
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        pending = _make_snippet(
            name="Re-met Pending",
            profile_url="/talent/profile/re-met-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        _seed_facial_yes(p, owner, pending)
        _track_pending(p, pending, owner)
        _contain_one(p, owner, progress)
        _settle_full(p, owner, pending, "REJECT")

        p.state.candidates._hydrate_resume_funnel_from_runtime(progress, store=p.state.context._runtime_state, run_id=p.state.context._runtime_run_id, brief_id=p.state.context._brief_id, stats=p.state.context.stats)

        key = p.state.candidates._funnel_candidate_key(pending)
        assert key not in p.state.candidates._resume_pending_full_decisions
        assert owner.full_reviewed_count == 1
        assert owner.full_reject_count == 1
        assert owner.full_outreach_count == 0
        assert p.state.context.stats["full_reviewed"] == 1
        assert p.state.context.stats["full_reject"] == 1

        # Control: no containment anywhere in the candidate's history.
        control = _make_pipeline(ctrl_td)
        control_owner = SearchString(
            id=11, name="owner", boolean="ml", status="done"
        )
        control_progress = Progress(
            brief_name="test", strings=[control_owner]
        )
        control.state.context._runtime_run_id, _ = control.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=control_progress,
        )
        control_snippet = _make_snippet(
            name="Re-met Pending",
            profile_url="/talent/profile/re-met-pending",
            source_string_id=control_owner.id,
            source_string_name=control_owner.name,
            page=1,
        )
        _seed_facial_yes(control, control_owner, control_snippet)
        _settle_full(control, control_owner, control_snippet, "REJECT")
        control.state.candidates._hydrate_resume_funnel_from_runtime(control_progress, store=control.state.context._runtime_state, run_id=control.state.context._runtime_run_id, brief_id=control.state.context._brief_id, stats=control.state.context.stats)

        assert control_owner.full_reviewed_count == owner.full_reviewed_count
        assert control_owner.full_reject_count == owner.full_reject_count
        assert control.state.context.stats["full_reviewed"] == p.state.context.stats["full_reviewed"]
        assert control.state.context.stats["full_reject"] == p.state.context.stats["full_reject"]

        # The SQL twin has to agree, or resume and health disagree about
        # whether this candidate still owes a review.
        _close_all_work_units(p)
        assert has_pending_work(p.state.context.output_dir) is False


def test_contained_row_yields_but_unactuated_save_remeet_stays_pending(monkeypatch):
    """Displacement must not swallow a SAVE that never reached LinkedIn.

    A save-family verdict with no succeeded save side effect is still owed work
    — the displacement rule has to hand the candidate back to the pending
    derivation, not settle it.
    """

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=12, name="owner", boolean="ml", status="done")
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        pending = _make_snippet(
            name="Unactuated Save",
            profile_url="/talent/profile/unactuated-save",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        _seed_facial_yes(p, owner, pending)
        _track_pending(p, pending, owner)
        _contain_one(p, owner, progress)
        _settle_full(p, owner, pending, "SAVE")

        p.state.candidates._hydrate_resume_funnel_from_runtime(progress, store=p.state.context._runtime_state, run_id=p.state.context._runtime_run_id, brief_id=p.state.context._brief_id, stats=p.state.context.stats)

        key = p.state.candidates._funnel_candidate_key(pending)
        assert key in p.state.candidates._resume_pending_full_decisions
        assert p.state.candidates._resume_pending_full_decisions[key] == "FACIAL_YES"
        assert p.state.candidates._resume_pending_full_owner_ids[key] == owner.id

        _close_all_work_units(p)
        assert has_pending_work(p.state.context.output_dir) is True


def test_recover_pending_full_contained_helper_failure_keeps_checkpoint_discipline(
    monkeypatch,
):
    """Containment blowing up must still leave the string persisted resumable."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=13,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        pending = _make_snippet(
            name="Exploding Skip",
            profile_url="/talent/profile/exploding-skip",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        key = _track_pending(p, pending, owner)
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.driver._abandon_unrecoverable_pending_full = MagicMock(
            side_effect=ValueError("canonical settle exploded")
        )
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()

        with pytest.raises(ValueError, match="canonical settle exploded"):
            asyncio.run(
                p.state.driver._recover_owner_pending_full_evaluations(
                    progress=progress,
                    search_string=owner,
                    first_incomplete_page=2,
                    string_stats=p.state.candidates._fresh_string_stats(),
                )
            )

        assert owner.status == "in_progress"
        p.state._allocator_service._checkpoint_progress.assert_called_once_with(progress)
        assert key in p.state.candidates._resume_pending_full_decisions


def test_recover_pending_full_contained_preserves_rendered_page(monkeypatch):
    """Pagination already advanced before containment; the cursor must survive."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=14,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=3,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        pending = _make_snippet(
            name="Page Two Pending",
            profile_url="/talent/profile/page-two-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=2,
        )
        _seed_facial_yes(p, owner, pending)
        _track_pending(p, pending, owner)
        p.state.surface._go_to_next_page_with_transient_retry = AsyncMock(
            return_value=(True, False)
        )
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()

        rendered_page = asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=3,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )

        assert rendered_page == 2
        p.state.surface._go_to_next_page_with_transient_retry.assert_awaited_once()
        assert p.state.candidates._resume_pending_full_decisions == {}


def _seed_failed_full_attempt(
    p,
    search_string: SearchString,
    snippet,
    error_text: str = (
        "Fireworks-cached response truncated: finish_reason=length"
    ),
) -> None:
    """Persist one non-succeeded full attempt — the trace a wedged evaluation leaves."""

    attempt_id = p.state.reviewer.attempts._start_runtime_stage_attempt(
        search_string=search_string,
        snippet=snippet,
        stage="full",
    )
    p.state.reviewer._finish_runtime_stage_failure(
        attempt_id=attempt_id,
        snippet=snippet,
        error=RuntimeError(error_text),
        stage="full",
    )


def test_recover_pending_full_exhausted_attempts_contained_before_reopen(
    monkeypatch,
    capsys,
):
    """CLO-147: two burned full attempts abandon the candidate BEFORE re-opening.

    The 2026-08-10/11 livelock re-opened the wedged candidate every resume (one
    governed open, instantly non-absorbable) and then crashed on the unsettled
    guard. With the budget exhausted, recovery must settle her with a receipt
    without navigating or opening anything."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=7,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        wedged = _make_snippet(
            name="Wedged Pending",
            profile_url="/talent/profile/wedged-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
            result_rank=1,
        )
        _seed_facial_yes(p, owner, wedged)
        _seed_failed_full_attempt(p, owner, wedged)
        _seed_failed_full_attempt(p, owner, wedged)
        key = _track_pending(p, wedged, owner)
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock()
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock()

        asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=2,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )

        p.state.context.browser.find_result_slot_by_profile_url.assert_not_awaited()
        p.state.reviewer._process_resumed_pending_full_evaluations.assert_not_awaited()
        assert key not in p.state.candidates._resume_pending_full_decisions
        assert key not in p.state.candidates._resume_pending_full_snippets
        assert key not in p.state.candidates._resume_pending_full_owner_ids
        payloads = _abandon_events(p)
        assert [payload["reason"] for payload in payloads] == [
            "recovery_attempts_exhausted"
        ]
        out = capsys.readouterr().out
        assert (
            "[recover-full] SKIPPED Wedged Pending — recovery_attempts_exhausted"
            in out
        )


def test_recover_pending_full_one_failed_attempt_still_recovers(monkeypatch):
    """One burned attempt is under the budget — recovery still gets its retry."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=8,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        once_failed = _make_snippet(
            name="Once Failed",
            profile_url="/talent/profile/once-failed",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
            result_rank=1,
        )
        _seed_facial_yes(p, owner, once_failed)
        _seed_failed_full_attempt(p, owner, once_failed)
        key = _track_pending(p, once_failed, owner)
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=2)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        recovered: list[str] = []

        async def settle_pending(**kwargs):
            snippet = kwargs["snippets"][0]
            recovered.append(snippet.profile_url)
            p.state.candidates.forget_pending(snippet.profile_url)
            return False

        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock(
            side_effect=settle_pending
        )

        asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=2,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )

        assert recovered == [once_failed.profile_url]
        assert key not in p.state.candidates._resume_pending_full_decisions
        assert _abandon_events(p) == []


def test_recover_pending_full_unsettled_second_failure_contains_in_session(
    monkeypatch,
    capsys,
):
    """The belt: an evaluation that completes without settling counts against
    the same budget in-session, so the second burned attempt abandons instead
    of raising the livelock."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=9,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        wedged = _make_snippet(
            name="Wedged In Session",
            profile_url="/talent/profile/wedged-in-session",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
            result_rank=1,
        )
        _seed_facial_yes(p, owner, wedged)
        _seed_failed_full_attempt(p, owner, wedged)
        key = _track_pending(p, wedged, owner)
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=3)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock(
            return_value=False
        )

        asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=2,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )

        assert key not in p.state.candidates._resume_pending_full_decisions
        payloads = _abandon_events(p)
        assert [payload["reason"] for payload in payloads] == [
            "recovery_attempts_exhausted"
        ]


def test_recover_pending_full_unsettled_fresh_candidate_still_raises(
    monkeypatch,
):
    """A fresh unsettled wedge (no burned attempts) still surfaces loudly: the
    first completed-but-unsettled evaluation is under the budget, so the
    unsettled guard keeps its raise."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=10,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        fresh = _make_snippet(
            name="Fresh Unsettled",
            profile_url="/talent/profile/fresh-unsettled",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
            result_rank=1,
        )
        _seed_facial_yes(p, owner, fresh)
        key = _track_pending(p, fresh, owner)
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=1)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock(
            return_value=False
        )

        with pytest.raises(
            RuntimeError,
            match="remains unsettled for active owner",
        ):
            asyncio.run(
                p.state.driver._recover_owner_pending_full_evaluations(
                    progress=progress,
                    search_string=owner,
                    first_incomplete_page=2,
                    string_stats=p.state.candidates._fresh_string_stats(),
                )
            )

        assert owner.status == "in_progress"
        assert key in p.state.candidates._resume_pending_full_decisions
        assert _abandon_events(p) == []


def test_recover_pending_full_run_level_aborts_do_not_count(monkeypatch):
    """Wave-2 review: run-level aborts write force_retryable=True — the
    process stopped, not the candidate's evaluation — and two of them must
    not abandon a healthy candidate."""

    monkeypatch.setattr(
        config,
        "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED",
        True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=11,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        healthy = _make_snippet(
            name="Healthy Aborted Twice",
            profile_url="/talent/profile/healthy-aborted-twice",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
            result_rank=1,
        )
        _seed_facial_yes(p, owner, healthy)
        for _ in range(2):
            attempt_id = p.state.reviewer.attempts._start_runtime_stage_attempt(
                search_string=owner,
                snippet=healthy,
                stage="full",
            )
            p.state.reviewer._abort_runtime_stage_attempt(
                attempt_id=attempt_id,
                snippet=healthy,
                error=RuntimeError("session expired mid-evaluation"),
            )
        key = _track_pending(p, healthy, owner)
        p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=2)
        p.state._allocator_service._checkpoint_progress = MagicMock()
        recovered: list[str] = []

        async def settle_pending(**kwargs):
            snippet = kwargs["snippets"][0]
            recovered.append(snippet.profile_url)
            p.state.candidates.forget_pending(snippet.profile_url)
            return False

        p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock(
            side_effect=settle_pending
        )

        asyncio.run(
            p.state.driver._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=2,
                string_stats=p.state.candidates._fresh_string_stats(),
            )
        )

        assert recovered == [healthy.profile_url]
        assert key not in p.state.candidates._resume_pending_full_decisions
        assert _abandon_events(p) == []


# ---------------------------------------------------------------------------
# Ahead-of-cursor pending (2026-08-14 CISO resume wedge, plans/resume-cursor-
# and-location-apply-hardening.md): a committed mutation (drift/variant) resets
# the cursor to page 1 of a NEW result set, but pending metadata carries no
# variant identity — so a pending full recorded on the replaced surface
# compares as "ahead of the active owner cursor" on every resume and the old
# unconditional raise killed four consecutive runs. Containment settles those
# snippets through the receipted abandonment writer; a stable (root) surface
# keeps the raise, because there ahead-of-cursor really is a cursor rollback.
# ---------------------------------------------------------------------------


def _experiment_state_stub(active_variant_id: str):
    from types import SimpleNamespace

    return SimpleNamespace(active_variant_id=active_variant_id)


def test_ahead_of_cursor_pending_raises_when_containment_disabled(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", False)

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=13, name="owner", boolean="ciso", status="done", pages_reviewed=2
        )
        progress = Progress(brief_name="test", strings=[owner])
        pending = _make_snippet(
            name="Ahead Pending",
            profile_url="/talent/profile/ahead-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=12,
            result_rank=25,
        )
        key = _track_pending(p, pending, owner)
        p.state._allocator_service._checkpoint_progress = MagicMock()

        with pytest.raises(
            RuntimeError, match="ahead of the active owner cursor"
        ):
            p.state.driver._resolve_ahead_of_cursor_pending(
                progress=progress,
                search_string=owner,
                experiment_state=_experiment_state_stub("drift-13-1"),
                owned_pending=[(key, pending)],
                first_incomplete_page=3,
            )

        assert owner.status == "in_progress"
        p.state._allocator_service._checkpoint_progress.assert_called_once_with(
            progress, search_string=owner
        )
        assert key in p.state.candidates._resume_pending_full_decisions


def test_ahead_of_cursor_pending_on_root_surface_still_raises(monkeypatch):
    """On a stable surface pages complete sequentially, so ahead-of-cursor
    pending means the cursor rolled back — the corruption tripwire survives
    containment being on."""

    monkeypatch.setattr(
        config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", True
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=13, name="owner", boolean="ciso", status="done", pages_reviewed=2
        )
        progress = Progress(brief_name="test", strings=[owner])
        pending = _make_snippet(
            name="Ahead Pending",
            profile_url="/talent/profile/ahead-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=12,
            result_rank=25,
        )
        key = _track_pending(p, pending, owner)
        p.state._allocator_service._checkpoint_progress = MagicMock()

        with pytest.raises(
            RuntimeError, match="ahead of the active owner cursor"
        ):
            p.state.driver._resolve_ahead_of_cursor_pending(
                progress=progress,
                search_string=owner,
                experiment_state=_experiment_state_stub("root"),
                owned_pending=[(key, pending)],
                first_incomplete_page=3,
            )

        assert owner.status == "in_progress"
        assert key in p.state.candidates._resume_pending_full_decisions


def test_ahead_of_cursor_pending_contained_after_surface_change(monkeypatch):
    """Drifted string + containment on: the ahead snippet settles with a
    receipt, the behind-cursor snippet stays queued for normal recovery."""

    monkeypatch.setattr(
        config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", True
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=13,
            name="owner",
            boolean="ciso",
            status="in_progress",
            pages_reviewed=2,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        ahead = _make_snippet(
            name="Ahead Pending",
            profile_url="/talent/profile/ahead-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=12,
            result_rank=25,
        )
        behind = _make_snippet(
            name="Behind Pending",
            profile_url="/talent/profile/behind-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=2,
            result_rank=3,
        )
        _seed_facial_yes(p, owner, ahead)
        _seed_facial_yes(p, owner, behind)
        ahead_key = _track_pending(p, ahead, owner)
        behind_key = _track_pending(p, behind, owner)

        remaining = p.state.driver._resolve_ahead_of_cursor_pending(
            progress=progress,
            search_string=owner,
            experiment_state=_experiment_state_stub("drift-13-1"),
            owned_pending=[(behind_key, behind), (ahead_key, ahead)],
            first_incomplete_page=3,
        )

        assert remaining == [(behind_key, behind)]
        events = _abandon_events(p)
        assert [event["reason"] for event in events] == [
            "surface_changed_ahead_of_cursor"
        ]
        assert events[0]["candidate_name"] == "Ahead Pending"
        assert events[0]["status"] == "failed"
        assert ahead_key not in p.state.candidates._resume_pending_full_decisions
        assert ahead_key not in p.state.candidates._resume_pending_full_snippets
        assert ahead_key not in p.state.candidates._resume_pending_full_owner_ids
        assert behind_key in p.state.candidates._resume_pending_full_decisions


def test_settle_owner_pending_before_surface_change_contains_owned_only(
    monkeypatch,
):
    """The mutation-boundary settle receipts THIS string's unsettled pending
    and leaves other strings' pending untouched."""

    monkeypatch.setattr(
        config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", True
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=13,
            name="owner",
            boolean="ciso",
            status="in_progress",
            pages_reviewed=12,
        )
        other = SearchString(
            id=7, name="other", boolean="ir", status="in_progress", pages_reviewed=1
        )
        progress = Progress(brief_name="test", strings=[owner, other])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        owned = _make_snippet(
            name="Owned Pending",
            profile_url="/talent/profile/owned-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=12,
            result_rank=25,
        )
        foreign = _make_snippet(
            name="Foreign Pending",
            profile_url="/talent/profile/foreign-pending",
            source_string_id=other.id,
            source_string_name=other.name,
            page=4,
            result_rank=2,
        )
        _seed_facial_yes(p, owner, owned)
        _seed_facial_yes(p, other, foreign)
        owned_key = _track_pending(p, owned, owner)
        foreign_key = _track_pending(p, foreign, other)

        p.state.driver._settle_owner_pending_full_before_surface_change(
            search_string=owner,
            reason="surface_replaced_by_drift",
        )

        events = _abandon_events(p)
        assert [event["reason"] for event in events] == [
            "surface_replaced_by_drift"
        ]
        assert events[0]["candidate_name"] == "Owned Pending"
        assert owned_key not in p.state.candidates._resume_pending_full_decisions
        assert owned_key not in p.state.candidates._resume_pending_full_snippets
        assert owned_key not in p.state.candidates._resume_pending_full_owner_ids
        assert foreign_key in p.state.candidates._resume_pending_full_decisions
        assert foreign_key in p.state.candidates._resume_pending_full_snippets


def test_settle_owner_pending_before_surface_change_flag_off_is_noop(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", False)

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=13,
            name="owner",
            boolean="ciso",
            status="in_progress",
            pages_reviewed=12,
        )
        pending = _make_snippet(
            name="Owned Pending",
            profile_url="/talent/profile/owned-pending",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=12,
            result_rank=25,
        )
        key = _track_pending(p, pending, owner)

        p.state.driver._settle_owner_pending_full_before_surface_change(
            search_string=owner,
            reason="surface_replaced_by_drift",
        )

        assert key in p.state.candidates._resume_pending_full_decisions
        assert _abandon_events(p) == []


def test_settle_owner_pending_drops_snippetless_entry_in_process_only(
    monkeypatch,
):
    """A pending entry with no stored snippet has no canonical identity to
    settle against; the boundary settle drops it in-process so it cannot
    re-wedge, and writes nothing."""

    monkeypatch.setattr(
        config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", True
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=13,
            name="owner",
            boolean="ciso",
            status="in_progress",
            pages_reviewed=12,
        )
        key = "snippetless-key"
        p.state.candidates._resume_pending_full_decisions[key] = "FACIAL_YES"
        p.state.candidates._resume_pending_full_owner_ids[key] = owner.id

        p.state.driver._settle_owner_pending_full_before_surface_change(
            search_string=owner,
            reason="surface_replaced_by_drift",
        )

        assert key not in p.state.candidates._resume_pending_full_decisions
        assert key not in p.state.candidates._resume_pending_full_owner_ids
        assert _abandon_events(p) == []
