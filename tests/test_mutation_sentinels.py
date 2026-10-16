"""Mutation sentinels: the twenty most important functions each keep a lock that can fail.

A green suite proves nothing about a function unless some test goes red when
that function stops doing its job. For each function below this file empties
the body in a subprocess, runs the one named lock, and requires it to fail.
If a lock is deleted, renamed, mocked out or weakened until it no longer
notices, this test goes red inside the default gate, in seconds.

The pairs were measured, not guessed: ``tools/mutation_survey.py`` emptied each
function in a hermetic copy and recorded which tests failed. The full survey
stays a tool, run on request; this is the part of its answer that must stay
true between runs. The same module is loaded as a pytest plugin in the
subprocess (``-p tests.test_mutation_sentinels``), where it applies one stub
per lock for the length of that lock's run.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
_RESULT_ENV = "MUTATION_SENTINELS_RESULT"

# (module, qualified name, the lock that must fail when it is emptied)
#
# Eighteen of these locks fail on an assertion. Two cannot: every caller of
# ``_perform_full_save`` and ``can_start_session`` unpacks the return value at
# once, so an emptied body crashes before any assertion is reached. Those two
# sentinels still catch a deleted, renamed or mocked-out lock, but not one whose
# assertions were quietly removed.
SENTINELS: tuple[tuple[str, str, str], ...] = (
    # The save path
    (
        "linkedin.side_effects",
        "LinkedInSideEffectsService.handle_save_decision",
        "tests/test_adapter_services.py::test_linkedin_already_present_counts_as_already_present_not_saved",
    ),
    (
        "linkedin.candidate_review",
        "CandidateReviewer._perform_full_save",
        "tests/test_full_eval_pipelining.py::test_serial_save_reuses_identity_verified_panel_and_one_governed_open",
    ),
    # Canonical runtime-state writers
    (
        "shared.runtime_state.store",
        "RuntimeStateStore.set_candidate_state",
        "tests/test_runtime_state.py::test_rejects_invalid_state_transition",
    ),
    (
        "shared.runtime_state.store",
        "RuntimeStateStore.finish_attempt_success",
        "tests/test_runtime_projections.py::test_stage_projections_exclude_failure_decisions_without_mutating_attempts",
    ),
    (
        "shared.runtime_state.store",
        "RuntimeStateStore.finish_attempt_failure",
        "tests/test_full_eval_pipelining.py::test_serial_full_cancellation_during_open_closes_attempt",
    ),
    (
        "shared.runtime_state.store",
        "RuntimeStateStore.begin_candidate_side_effect",
        "tests/test_runtime_state.py::test_reconciles_pending_candidate_side_effects_and_allows_manual_replay",
    ),
    (
        "shared.runtime_state.store",
        "RuntimeStateStore.complete_candidate_side_effect",
        "tests/test_adapter_services.py::test_cancelled_save_blocks_new_dispatch_and_books_landed_save[dispatch]",
    ),
    (
        "shared.runtime_state.store",
        "RuntimeStateStore.list_terminal_identity_keys",
        "tests/test_runtime_state.py::test_suppression_is_brief_version_aware",
    ),
    (
        "shared.runtime_state.store",
        "RuntimeStateStore.reconcile_open_attempts",
        "tests/test_runtime_state.py::test_reconciles_orphaned_attempts_on_startup",
    ),
    # Evaluation-loop stages
    (
        "linkedin.evaluation_loop",
        "EvaluationLoop.run_full",
        "tests/test_linkedin_pipeline.py::test_run_full_resume_never_executes_foreign_pending_before_active_owner",
    ),
    (
        "linkedin.candidate_review",
        "CandidateReviewer._full_evaluate",
        "tests/test_full_eval_pipelining.py::test_serial_full_tool_provider_failure_propagates_and_closes_attempt",
    ),
    (
        "linkedin.page_review",
        "PageReviewer._review_page_batch",
        "tests/test_facial_sniper_prefix.py::test_batch_facial_threads_the_constructed_prefix",
    ),
    (
        "linkedin.page_review",
        "PageReviewer._review_page_sequentially",
        "tests/test_linkedin_pipeline.py::test_page_raises_after_five_consecutive_card_extraction_failures",
    ),
    (
        "linkedin.string_driver",
        "StringDriver._process_string_impl",
        "tests/test_linkedin_zero_results_signal.py::test_zero_results_after_prior_results_classifies_as_suspicious",
    ),
    # Governor
    (
        "shared.governor",
        "SessionGovernor.check_profile_open_or_raise",
        "tests/test_panel_wedge_containment.py::test_force_backoff_threads_custom_cooldown",
    ),
    (
        "shared.governor",
        "SessionGovernor.can_start_session",
        "tests/test_cooldown.py::test_governor_can_start_session_honors_persisted_backoff",
    ),
    # Geography gate
    (
        "linkedin.geography_gate",
        "GeographyGateService._apply_session_location_filter",
        "tests/test_orchestrator_geography_gate.py::test_successful_apply_sets_flag_and_splits_semicolon_values",
    ),
    (
        "linkedin.geography_gate",
        "GeographyGateService._verify_session_geography_chips",
        "tests/test_orchestrator_geography_gate.py::test_missing_chips_after_reassert_fail_closed",
    ),
    # Judgment parsing
    (
        "shared.judgment.templates",
        "parse_full_evaluation_response",
        "tests/test_confidence_provenance.py::test_missing_confidence_line_returns_none_with_flag",
    ),
    (
        "shared.judgment.templates",
        "parse_facial_response",
        "tests/test_judgment_decision_anchoring.py::test_facial_clean_no_resolves_to_facial_no",
    ),
)


# ---------------------------------------------------------------------------
# Emptying a function in process
# ---------------------------------------------------------------------------


def _empty_like(original: Any) -> Any:
    """A stub of the same kind as ``original`` that does nothing."""
    if inspect.isasyncgenfunction(original):

        async def empty_async_generator(*args: Any, **kwargs: Any):
            return
            yield  # pragma: no cover

        return empty_async_generator
    if inspect.iscoroutinefunction(original):

        async def empty_coroutine(*args: Any, **kwargs: Any) -> None:
            return None

        return empty_coroutine
    if inspect.isgeneratorfunction(original):

        def empty_generator(*args: Any, **kwargs: Any):
            return
            yield  # pragma: no cover

        return empty_generator

    def empty(*args: Any, **kwargs: Any) -> None:
        return None

    return empty


def empty_function(module_name: str, qualname: str) -> list[tuple[Any, str, Any]]:
    """Replace the function with an empty stub everywhere it is bound; return the undo list."""
    module = importlib.import_module(module_name)
    *owners, name = qualname.split(".")
    owner: Any = module
    for part in owners:
        owner = getattr(owner, part)
    raw = vars(owner)[name]
    original = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
    stub: Any = _empty_like(original)
    if isinstance(raw, staticmethod):
        stub = staticmethod(stub)
    elif isinstance(raw, classmethod):
        stub = classmethod(stub)
    undo: list[tuple[Any, str, Any]] = [(owner, name, raw)]
    setattr(owner, name, stub)
    if not owners:
        # ``from module import name`` leaves other bindings to the original.
        for other in list(sys.modules.values()):
            if other is None or other is module:
                continue
            for attribute, value in list(getattr(other, "__dict__", {}).items()):
                if value is original:
                    undo.append((other, attribute, value))
                    setattr(other, attribute, stub)
    return undo


def restore(undo: list[tuple[Any, str, Any]]) -> None:
    for owner, name, value in reversed(undo):
        setattr(owner, name, value)


# ---------------------------------------------------------------------------
# Plugin half: active only in the subprocess, where this module is loaded with -p
# ---------------------------------------------------------------------------

_LOCK_TARGETS = {lock: (module, qualname) for module, qualname, lock in SENTINELS}
_outcomes: dict[str, str] = {}


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    # The subprocess is given whole files, so a deleted or renamed lock shows
    # up as the one lock that did not run instead of aborting the whole session.
    if os.environ.get(_RESULT_ENV):
        dropped = [item for item in items if item.nodeid not in _LOCK_TARGETS]
        items[:] = [item for item in items if item.nodeid in _LOCK_TARGETS]
        if dropped:
            config.hook.pytest_deselected(items=dropped)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: Any, nextitem: Any) -> Iterator[None]:
    target = _LOCK_TARGETS.get(item.nodeid) if os.environ.get(_RESULT_ENV) else None
    undo = empty_function(*target) if target else []
    try:
        yield
    finally:
        restore(undo)


def pytest_runtest_logreport(report: Any) -> None:
    if os.environ.get(_RESULT_ENV) and (report.failed or report.when == "call"):
        if _outcomes.get(report.nodeid) != "failed":
            _outcomes[report.nodeid] = "failed" if report.failed else report.outcome


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    result_path = os.environ.get(_RESULT_ENV)
    if result_path:
        Path(result_path).write_text(json.dumps(_outcomes), encoding="utf-8")


# ---------------------------------------------------------------------------
# The sentinel
# ---------------------------------------------------------------------------


def _run_locks_with_their_functions_emptied(result_path: Path) -> dict[str, str]:
    env = {**os.environ, _RESULT_ENV: str(result_path), "PYTHONPATH": str(ROOT)}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.test_mutation_sentinels",
            *sorted({lock.split("::", 1)[0] for _, _, lock in SENTINELS}),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if not result_path.exists():
        pytest.fail("the sentinel subprocess wrote no result:\n" + (done.stdout + done.stderr)[-2000:])
    return json.loads(result_path.read_text(encoding="utf-8"))


def test_every_sentinel_lock_fails_when_its_function_is_emptied(tmp_path):
    outcomes = _run_locks_with_their_functions_emptied(tmp_path / "sentinels.json")

    still_green = [
        f"{module}:{qualname} <- {lock} ({outcomes.get(lock, 'did not run')})"
        for module, qualname, lock in SENTINELS
        if outcomes.get(lock) != "failed"
    ]
    assert not still_green, "locks that no longer fail when their function is emptied:\n" + "\n".join(
        still_green
    )


@pytest.mark.structural
def test_the_sentinel_list_is_twenty_distinct_functions_with_distinct_locks():
    assert len(SENTINELS) == 20
    assert len({(module, qualname) for module, qualname, _ in SENTINELS}) == 20
    assert len({lock for _, _, lock in SENTINELS}) == 20


def test_emptying_is_undone_and_reaches_names_imported_elsewhere():
    from shared import judger
    from shared.judgment import templates

    original = templates.parse_facial_response
    assert judger.parse_facial_response is original

    undo = empty_function("shared.judgment.templates", "parse_facial_response")
    try:
        assert templates.parse_facial_response("DECISION: FACIAL_YES") is None
        assert judger.parse_facial_response is templates.parse_facial_response
    finally:
        restore(undo)

    assert templates.parse_facial_response is original
    assert judger.parse_facial_response is original
