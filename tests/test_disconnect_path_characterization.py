"""Bounded disconnect recovery re-enters from canonical runtime state."""

import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from shared.schemas import CandidateSnippet, OpusDecision, Progress, SearchString
from shared.safety import RunStopReason


_PROJECT_SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
)


def _orchestrator_mod():
    """Resolve the module at call time because adapter tests reload it."""

    return importlib.import_module("linkedin.orchestrator")


def _make_pipeline(tmp_path: Path):
    orch = _orchestrator_mod()
    with (
        patch.object(orch, "load_brief") as load_brief,
        patch.object(orch, "init_judger"),
        patch.object(orch, "LinkedInBrowser"),
    ):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.kit_url = ""
        brief.permanent_filters = {}
        brief.needs_preflight.return_value = False
        load_brief.return_value = brief

        brief_path = tmp_path / "brief.json"
        brief_path.write_text('{"id": "test"}')
        return orch.Pipeline(
            brief_path=str(brief_path),
            output_dir=str(tmp_path),
        )


@pytest.fixture(scope="module")
def disconnect_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("disconnect-characterization")
    pipeline = _make_pipeline(tmp_path)
    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(
                id=5,
                name="interrupted",
                boolean="one",
                status="in_progress",
                block="Block A",
                pages_reviewed=1,
            )
        ],
        current_string_id=5,
        current_page=1,
    )
    pipeline._runtime_run_id, _ = pipeline._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )

    pipeline.test_mode = False
    pipeline.browser.connect = AsyncMock()
    pipeline.browser.disconnect = AsyncMock()
    pipeline.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
    pipeline.browser.current_profile_identity_fragment.return_value = "ada"
    failure = RuntimeError("Page.evaluate: Target crashed")
    save_landed = {"value": False}

    async def ambiguous_save(**_kwargs):
        save_landed["value"] = True
        raise failure

    pipeline.browser.is_already_saved_on_card = AsyncMock(
        side_effect=lambda *_args: save_landed["value"]
    )
    pipeline.browser.save_candidate = AsyncMock(side_effect=ambiguous_save)
    pipeline.browser.scroll_for_linger = AsyncMock(return_value=0)
    pipeline.browser.scroll_restore = AsyncMock()
    recovery_snapshot = MagicMock(name="recovery_snapshot")
    order = []
    pipeline._capture_recovery_snapshot = AsyncMock(
        side_effect=lambda *_args, **_kwargs: (
            order.append("capture") or recovery_snapshot
        )
    )
    pipeline._recovery_service.recover = AsyncMock(
        side_effect=lambda **_kwargs: order.append("recover") or True
    )
    pipeline._reassert_session_location_after_recovery = AsyncMock(
        side_effect=lambda: order.append("reassert")
    )
    pipeline._run_block_adaptation = AsyncMock()
    pipeline._print_session_summary = MagicMock()
    pipeline._print_summary = MagicMock()
    pipeline._generate_run_report = MagicMock()
    pipeline._finalize_run_snapshot = MagicMock(
        return_value=tmp_path / "frozen-run"
    )
    pipeline._session_expired = MagicMock()
    pipeline._session_expired.is_set.return_value = False

    checkpoint_progress = MagicMock(wraps=pipeline._checkpoint_progress)
    pipeline._checkpoint_progress = checkpoint_progress
    incident = {"calls": []}
    attempt_id = None

    async def save_then_disconnect(search_string, live_progress):
        nonlocal attempt_id
        call_number = len(incident["calls"]) + 1
        experiment_state = pipeline._experiment_state_for(search_string)
        incident["calls"].append(
            {
                "progress": live_progress,
                "search_string": search_string,
                "values": (
                    live_progress.current_string_id,
                    live_progress.current_page,
                    search_string.boolean,
                    search_string.pages_reviewed,
                    experiment_state.active_allocator_page_cursor(),
                ),
            }
        )
        order.append(f"process:{call_number}")
        if call_number == 2:
            attempt_id = pipeline._start_runtime_stage_attempt(
                search_string=search_string,
                snippet=incident["snippet"],
                stage="full",
            )
            outcome = await pipeline._side_effects_service.handle_save_decision(
                snippet=incident["snippet"],
                runtime_search_string=search_string,
                attempt_id=attempt_id,
            )
            pipeline._finish_runtime_stage_success(
                attempt_id=attempt_id,
                stage="full",
                snippet=incident["snippet"],
                decision=incident["decision"],
            )
            search_string.saves.append(incident["snippet"].name)
            incident["replay_outcome"] = outcome
            return

        live_progress.current_page = 2
        experiment_state.set_active_allocator_page_cursor(2)
        pipeline._checkpoint_progress(
            live_progress,
            search_string=search_string,
            page_num=2,
        )
        pipeline._arm_incomplete_page_rollback(search_string, experiment_state)
        snippet = CandidateSnippet(
            name="Ada Lovelace",
            headline="ML Engineer",
            current_title="ML Engineer",
            current_company="Analytical Engines",
            location="London",
            education_snippet="",
            profile_url="/talent/profile/ada",
            source_string_id=search_string.id,
            source_string_name=search_string.name,
            page=2,
            result_rank=1,
        )
        incident["snippet"] = snippet
        pipeline._record_runtime_snippet(search_string, snippet)
        attempt_id = pipeline._start_runtime_stage_attempt(
            search_string=search_string,
            snippet=snippet,
            stage="full",
        )
        incident["decision"] = OpusDecision(
            stage="full",
            decision="SAVE",
            path="DIRECT:test",
            confidence=0.95,
            rationale="Strong direct fit",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        pipeline.stats["save_attempts"] = 1
        search_string.boolean = "mutated after checkpoint"
        search_string.pages_reviewed = 99
        experiment_state.set_active_allocator_page_cursor(99)
        try:
            await pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=search_string,
                attempt_id=attempt_id,
            )
        except BaseException as exc:
            pipeline._abort_runtime_stage_attempt(
                attempt_id=attempt_id,
                snippet=snippet,
                error=exc,
                payload={"stage": "full"},
            )
            raise
        raise AssertionError("ambiguous save must interrupt confirmation")

    pipeline._process_string_impl = save_then_disconnect

    with (
        patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_SECONDS", 0),
        patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_SECONDS", 0),
        patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_BASE_SECONDS", 0),
    ):
        asyncio.run(pipeline.run_full(resume=True))

    run_id = pipeline._runtime_run_id
    canonical_progress = pipeline._runtime_bridge.load_progress(run_id)
    experiment_states = pipeline._runtime_bridge.load_experiment_states(
        run_id,
        progress=canonical_progress,
    )
    work_units = pipeline._runtime_state.list_work_units(
        run_id,
        kind="linkedin_string",
    )
    with pipeline._runtime_state.connect() as conn:
        run = dict(
            conn.execute(
                "SELECT status, stop_reason FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        )
        attempt_count = conn.execute(
            "SELECT COUNT(*) AS count FROM candidate_attempts ca "
            "JOIN candidates c ON c.id = ca.candidate_id "
            "WHERE c.identity_key = ? AND ca.stage = 'full'",
            ("/talent/profile/ada",),
        ).fetchone()["count"]
    side_effects = pipeline._runtime_state.list_candidate_side_effects(
        source="linkedin",
        brief_id="test-project",
        identity_key="/talent/profile/ada",
    )

    return {
        "order": order,
        "snapshot": recovery_snapshot,
        "capture": pipeline._capture_recovery_snapshot,
        "recovery": pipeline._recovery_service.recover,
        "reassert": pipeline._reassert_session_location_after_recovery,
        "run_id": run_id,
        "run": run,
        "attempt_count": attempt_count,
        "side_effects": side_effects,
        "save_candidate": pipeline.browser.save_candidate,
        "checkpoint_progress": checkpoint_progress,
        "incident": incident,
        "progress": canonical_progress,
        "experiment_state": experiment_states[5],
        "work_units": work_units,
    }


def test_recovered_disconnect_reenters_same_string_from_canonical_state(
    disconnect_result,
):
    assert disconnect_result["order"] == [
        "process:1",
        "capture",
        "recover",
        "reassert",
        "process:2",
    ]
    assert disconnect_result["run"] == {
        "status": "completed",
        "stop_reason": RunStopReason.NORMAL,
    }
    first, second = disconnect_result["incident"]["calls"]
    assert second["progress"] is not first["progress"]
    assert second["search_string"] is not first["search_string"]
    assert second["values"] == (5, 2, "one", 2, 2)
    checkpoint_call = call(
        first["progress"],
        search_string=first["search_string"],
        page_num=2,
    )
    assert (
        disconnect_result["checkpoint_progress"].call_args_list.count(
            checkpoint_call
        )
        == 2
    )
    disconnect_result["capture"].assert_awaited_once_with(
        first["search_string"],
        page_num=2,
    )
    disconnect_result["recovery"].assert_awaited_once_with(
        run_id=disconnect_result["run_id"],
        snapshot=disconnect_result["snapshot"],
    )
    disconnect_result["reassert"].assert_awaited_once_with()


def test_ambiguous_landed_save_is_reconciled_without_second_click(
    disconnect_result,
):
    assert disconnect_result["attempt_count"] == 2
    assert len(disconnect_result["side_effects"]) == 1
    side_effect = disconnect_result["side_effects"][0]
    assert side_effect["status"] == "succeeded"
    assert side_effect["attempt_count"] == 2
    replay_outcome = disconnect_result["incident"]["replay_outcome"]
    assert replay_outcome.status == "succeeded"
    assert replay_outcome.payload["already_present"] is True
    disconnect_result["save_candidate"].assert_awaited_once()


def test_recovery_preserves_canonical_page_checkpoint(
    disconnect_result,
):
    progress = disconnect_result["progress"]
    assert progress.current_string_id == 5
    assert progress.current_page == 2
    assert progress.strings[0].boolean == "one"
    assert disconnect_result["experiment_state"].active_allocator_page_cursor() == 2
    assert [
        (item.id, item.status, item.pages_reviewed)
        for item in progress.strings
    ] == [(5, "done", 2)]
    assert [
        (unit["source_unit_id"], unit["status"])
        for unit in disconnect_result["work_units"]
    ] == [("5", "done")]
