"""Search memory counts a string once per resume lineage, not once per clone.

Every resumed session clones the previous run's work units with their status
and counters (``RuntimeStateStore.start_run`` with
``clone_work_units_from_run_id``), and production always links the new run to
the one it cloned (``resumed_from_run_id``, shared/runtime_state/
linkedin_run_start.py). A string finished in one run therefore reappears as a
``done`` row in every later run of the chain. The SPL SWE campaign (project
9000000003) finished string 1 in run 9 and was resumed six more times, so
strategy formation read 7 x 80 = 560 saves over 7 x 23 = 161 pages for a
string that saved 80 people over 23 pages.

The projection keeps the terminal copy of each lineage: the one no later run
in the chain carries forward. That copy holds the ancestry-hydrated
cumulative counters. Separate fresh runs are separate lineages and still
count separately.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from shared.runtime_state.projections import project_linkedin_search_memory
from shared.runtime_state.store import LINKEDIN_STRING_KIND, RuntimeStateStore
from shared.schemas import SearchString

BRIEF_ID = "brief-lineage"
SAVES = [f"saved-{i}" for i in range(80)]
PAGES = 23
CANDIDATES = 577


def _start_run(store: RuntimeStateStore, *, resumed_from: int | None = None) -> int:
    return store.start_run(
        source="linkedin",
        brief_id=BRIEF_ID,
        output_dir=str(store.db_path.parent),
        mode="resume" if resumed_from else "fresh",
        resume_state={"brief_name": BRIEF_ID},
        # The production resume shape: link and clone from the same run.
        resumed_from_run_id=resumed_from,
        clone_work_units_from_run_id=resumed_from,
    )


def _write_string(
    store: RuntimeStateStore,
    *,
    run_id: int,
    status: str,
    source_unit_id: str = "1",
) -> int:
    string = SearchString(
        id=int(source_unit_id),
        name=f"string-{source_unit_id}",
        boolean="python",
        status=status,
        pages_reviewed=PAGES,
        saves=list(SAVES),
        family_key="builders",
    )
    return store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        kind=LINKEDIN_STRING_KIND,
        source_unit_id=source_unit_id,
        display_name=string.name,
        ordering_index=int(source_unit_id),
        status=status,
        payload=string.to_dict(),
        checkpoint={"pages_reviewed": PAGES},
        metrics={"pages_reviewed": PAGES},
        family_key=string.family_key,
        counters={"candidates_discovered": CANDIDATES},
    )


def _execute_string(store: RuntimeStateStore, *, run_id: int, source_unit_id: str = "1") -> int:
    _write_string(store, run_id=run_id, status="in_progress", source_unit_id=source_unit_id)
    return _write_string(store, run_id=run_id, status="done", source_unit_id=source_unit_id)


def _project(store: RuntimeStateStore, *, flag: bool, run_id: int | None = None) -> dict:
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", flag):
        return project_linkedin_search_memory(store, brief_id=BRIEF_ID, run_id=run_id)


@pytest.fixture()
def store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def test_a_chained_resume_counts_the_finished_string_once(store) -> None:
    fresh = _start_run(store)
    _execute_string(store, run_id=fresh)
    resumed = _start_run(store, resumed_from=fresh)
    _start_run(store, resumed_from=resumed)

    overall = _project(store, flag=False)["overall"]

    assert overall["strings_seen"] == 1
    assert overall["saves"] == len(SAVES)
    assert overall["pages_reviewed"] == PAGES
    assert overall["candidates_seen"] == CANDIDATES


def test_the_spl_shape_six_dead_resumes_after_the_finish_stay_one_string(store) -> None:
    """Run 9 finished string 1; runs 10-15 each re-cloned it and died."""

    run_id = _start_run(store)
    _execute_string(store, run_id=run_id)
    for _ in range(6):
        run_id = _start_run(store, resumed_from=run_id)

    overall = _project(store, flag=False)["overall"]

    assert overall["saves"] == 80, "the clone multiplier is back: 7 x 80 = 560"
    assert overall["pages_reviewed"] == 23, "the clone multiplier is back: 7 x 23 = 161"
    assert overall["strings_seen"] == 1


def test_two_fresh_runs_of_the_same_string_are_two_lineages(store) -> None:
    first = _start_run(store)
    _execute_string(store, run_id=first)
    second = _start_run(store)
    _execute_string(store, run_id=second)

    overall = _project(store, flag=False)["overall"]

    assert overall["strings_seen"] == 2
    assert overall["saves"] == 2 * len(SAVES)


def test_a_string_finished_after_a_resume_counts_once(store) -> None:
    """In progress when the session ended, finished in the resumed run."""

    fresh = _start_run(store)
    _write_string(store, run_id=fresh, status="in_progress")
    resumed = _start_run(store, resumed_from=fresh)
    _execute_string(store, run_id=resumed)
    _start_run(store, resumed_from=resumed)

    assert _project(store, flag=False)["overall"]["strings_seen"] == 1


def test_under_the_contamination_flag_the_executed_row_still_counts(store) -> None:
    """Flag on, clones (never started) are invalid, so the executed row is the
    only valid copy and must not be dropped as superseded by an invalid one."""

    fresh = _start_run(store)
    _execute_string(store, run_id=fresh)
    resumed = _start_run(store, resumed_from=fresh)
    _start_run(store, resumed_from=resumed)

    overall = _project(store, flag=True)["overall"]

    assert overall["strings_seen"] == 1
    assert overall["saves"] == len(SAVES)


def test_a_run_scoped_projection_still_returns_that_runs_rows(store) -> None:
    fresh = _start_run(store)
    _execute_string(store, run_id=fresh)
    _start_run(store, resumed_from=fresh)

    scoped = _project(store, flag=False, run_id=fresh)

    assert scoped["overall"]["strings_seen"] == 1
