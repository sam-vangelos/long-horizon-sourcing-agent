"""Exa source module S5 — the module-owned runtime-state bridge.

A real :class:`~shared.runtime_state.store.RuntimeStateStore` in
``tmp_path``, never a mock: the whole point of the bridge is the shape of
the rows it writes, and a mock would assert the shape this file already
believes. Every assertion here reads the store back.

The load direction is the one that matters. A resumed run rebuilds its
query list from the work-unit ROWS, not from ``progress.json`` — the file
is a compatibility projection, and the one moment it can disagree with
SQLite is a crash, which is the one moment resume runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from exa.runtime_bridge import ExaRuntimeStateBridge
from exa.schemas import ExaProgress, ExaSearchQuery
from shared.runtime_state.store import EXA_QUERY_KIND, RuntimeStateStore


BRIEF_ID = "cos-exa"
BRIEF_NAME = "Chief of Staff to the CEO"


def _brief_file(tmp_path: Path) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps({"role_title": BRIEF_NAME, "version": "1"}))
    return path


def _bridge(tmp_path: Path) -> tuple[RuntimeStateStore, ExaRuntimeStateBridge]:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    bridge = ExaRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=BRIEF_ID,
        brief_name=BRIEF_NAME,
        brief_path=str(_brief_file(tmp_path)),
    )
    return store, bridge


def _queries() -> list[ExaSearchQuery]:
    return [
        ExaSearchQuery(
            id=1,
            name="people_search · Benchmark",
            lane="people_search",
            firm="Benchmark",
            query="Senior Associate at Benchmark",
        ),
        ExaSearchQuery(
            id=2,
            name="team_page · Benchmark",
            lane="team_page",
            firm="Benchmark",
            query="Benchmark team page",
        ),
    ]


def _mark_terminal(
    store: RuntimeStateStore,
    *,
    run_id: int,
    identity_key: str,
    decision: str = "SAVE",
) -> None:
    """Walk one candidate to a terminal verdict through the real guards."""

    store.ensure_candidate(
        source="exa",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        display_name="Avery Holt",
        person_key=f"exa:{identity_key}",
    )
    for state in ("snippet_extracted", "full_started"):
        store.set_candidate_state(
            run_id=run_id,
            source="exa",
            brief_id=BRIEF_ID,
            identity_key=identity_key,
            new_state=state,
        )
    store.set_candidate_state(
        run_id=run_id,
        source="exa",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        new_state="full_terminal",
        terminal_decision=decision,
    )


# ---------------------------------------------------------------------------


def test_fresh_run_writes_query_work_units(tmp_path: Path) -> None:
    store, bridge = _bridge(tmp_path)

    run_id, progress = bridge.start_or_resume_run(
        resume=False,
        initial_progress=ExaProgress(brief_name=BRIEF_NAME, queries=_queries()),
    )

    rows = store.list_work_units(run_id, kind=EXA_QUERY_KIND)
    assert [row["source_unit_id"] for row in rows] == ["1", "2"]
    assert [row["status"] for row in rows] == ["queued", "queued"]
    assert [row["domain_lane"] for row in rows] == ["people_search", "team_page"]
    assert [row["family_key"] for row in rows] == ["Benchmark", "Benchmark"]
    assert rows[0]["display_name"] == "people_search · Benchmark"
    # The payload round-trips the compiled query, so a resumed run does
    # not have to recompile from a brief that may have moved.
    payload = json.loads(rows[0]["payload_json"])
    assert ExaSearchQuery.from_dict(payload).query == "Senior Associate at Benchmark"
    assert len(progress.queries) == 2

    run = store.get_run(run_id)
    assert run["source"] == "exa"
    assert run["mode"] == "fresh"
    # Brief identity is pinned so run review can tell which brief ran.
    assert run["brief_content_hash"]

    # The projection is written, and it is derived, not authoritative.
    projected = json.loads((tmp_path / "state" / "progress.json").read_text())
    assert [q["id"] for q in projected["queries"]] == [1, 2]


def test_resume_rebuilds_progress_from_work_units(tmp_path: Path) -> None:
    store, bridge = _bridge(tmp_path)
    queries = _queries()
    first_run, progress = bridge.start_or_resume_run(
        resume=False,
        initial_progress=ExaProgress(brief_name=BRIEF_NAME, queries=queries),
    )
    progress.queries[0].status = "done"
    progress.queries[0].result_count = 17
    progress.queries[0].candidates_discovered = 4
    progress.queries[0].saves.append("linkedin.com/in/averylinholt")
    bridge.sync_progress(first_run, progress)

    # A stale projection on disk must not be what resume reads.
    (tmp_path / "state" / "progress.json").write_text(
        json.dumps({"brief_name": BRIEF_NAME, "queries": []})
    )

    second_run, resumed = bridge.start_or_resume_run(resume=True)

    assert second_run != first_run
    assert store.get_run(second_run)["mode"] == "resume"
    assert [q.id for q in resumed.queries] == [1, 2]
    assert [q.status for q in resumed.queries] == ["done", "queued"]
    assert resumed.queries[0].result_count == 17
    assert resumed.queries[0].candidates_discovered == 4
    assert resumed.queries[0].saves == ["linkedin.com/in/averylinholt"]
    # The compiled query text survives the round trip.
    assert resumed.queries[1].query == "Benchmark team page"


def test_resume_carries_terminal_identity_keys_as_seen(tmp_path: Path) -> None:
    store, bridge = _bridge(tmp_path)
    first_run, progress = bridge.start_or_resume_run(
        resume=False,
        initial_progress=ExaProgress(brief_name=BRIEF_NAME, queries=_queries()),
    )
    _mark_terminal(
        store,
        run_id=first_run,
        identity_key="linkedin.com/in/averylinholt",
        decision="SAVE",
    )

    _, resumed = bridge.start_or_resume_run(resume=True)

    assert "linkedin.com/in/averylinholt" in resumed.discovered_identity_keys


def test_sync_progress_deletes_missing_units(tmp_path: Path) -> None:
    store, bridge = _bridge(tmp_path)
    run_id, progress = bridge.start_or_resume_run(
        resume=False,
        initial_progress=ExaProgress(brief_name=BRIEF_NAME, queries=_queries()),
    )
    assert len(store.list_work_units(run_id, kind=EXA_QUERY_KIND)) == 2

    # The brief lost a firm: the compiled list shrinks, and the row for
    # the query that no longer exists must go with it.
    progress.queries = progress.queries[:1]
    bridge.sync_progress(run_id, progress)

    rows = store.list_work_units(run_id, kind=EXA_QUERY_KIND)
    assert [row["source_unit_id"] for row in rows] == ["1"]


# ---------------------------------------------------------------------------
# S8c — the brief is the authority on what runs
# ---------------------------------------------------------------------------


def _plan(*queries: ExaSearchQuery) -> ExaProgress:
    return ExaProgress(brief_name=BRIEF_NAME, queries=list(queries))


def _firm_query(query_id: int, firm: str, lane: str = "people_search") -> ExaSearchQuery:
    return ExaSearchQuery(
        id=query_id,
        name=f"{lane} · {firm}",
        lane=lane,
        firm=firm,
        query=f"Senior Associate at {firm}",
    )


def _archetype_query(query_id: int, archetype_id: str) -> ExaSearchQuery:
    return ExaSearchQuery(
        id=query_id,
        name=f"archetype · {archetype_id}",
        lane="archetype",
        firm="",
        archetype_id=archetype_id,
        query=f"Bay Area people matching {archetype_id}.",
        num_results=25,
        include_domains=["linkedin.com"],
    )


def _statuses(store: RuntimeStateStore, run_id: int) -> dict[str, str]:
    return {
        row["source_unit_id"]: row["status"]
        for row in store.list_work_units(run_id, kind=EXA_QUERY_KIND)
    }


def test_resume_appends_new_queries_from_revised_brief(
    tmp_path: Path, capsys
) -> None:
    """A lane added mid-campaign actually runs.

    The defect this locks: resume cloned the previous run's work units
    and returned THOSE, so the freshly compiled plan was computed and
    thrown away. Six archetype queries added to the brief on Tuesday
    were invisible to every session after it — the campaign ran the
    plan it was launched with, forever.
    """

    store, bridge = _bridge(tmp_path)
    session_one = _plan(_firm_query(1, "Benchmark"), _firm_query(2, "Greylock"))
    first_run, progress = bridge.start_or_resume_run(
        resume=False, initial_progress=session_one
    )
    progress.queries[0].status = "done"
    progress.queries[0].result_count = 11
    progress.queries[0].candidates_discovered = 3
    bridge.sync_progress(first_run, progress)
    capsys.readouterr()

    revised = _plan(
        _firm_query(1, "Benchmark"),
        _firm_query(2, "Greylock"),
        _archetype_query(3, "a01-mbb-to-vc"),
    )
    second_run, resumed = bridge.start_or_resume_run(
        resume=True,
        initial_progress=revised,
        active_lanes=("people_search", "archetype"),
    )

    rows = store.list_work_units(second_run, kind=EXA_QUERY_KIND)
    assert [row["source_unit_id"] for row in rows] == ["1", "2", "3"]
    assert [row["status"] for row in rows] == ["done", "queued", "queued"]
    # The firm units are untouched — same ids, same counters, same
    # verdicts. A merge that renumbered them would re-run a paid query.
    assert rows[0]["result_count"] == 11
    assert rows[0]["candidates_discovered"] == 3
    assert rows[2]["domain_lane"] == "archetype"
    assert rows[2]["family_key"] == "archetype:a01-mbb-to-vc"
    added = next(q for q in resumed.queries if q.lane == "archetype")
    assert added.id == 3 and added.status == "queued"
    assert added.num_results == 25

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "plan revised" in line]
    assert lines == [
        "    [exa] plan revised — 1 new query added: archetype · a01-mbb-to-vc"
    ]


def test_resume_appends_after_the_highest_stored_id_not_the_compiled_one(
    tmp_path: Path,
) -> None:
    """New ids continue the campaign's numbering, not the compiler's.

    Compiled ids are POSITIONAL: dropping a lane renumbers every query
    after it. A merge that trusted them would hand an archetype query
    the id of a firm query that already ran — same ``source_unit_id``,
    different search, and the campaign would report one as the other.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(*[_firm_query(i, f"Firm {i}") for i in range(1, 6)])
    first_run, progress = bridge.start_or_resume_run(
        resume=False, initial_progress=stored
    )
    for query in progress.queries:
        query.status = "done"
    bridge.sync_progress(first_run, progress)

    # The revised brief keeps the five firms and adds two archetypes,
    # which the compiler numbers 6 and 7 — but it also renumbers them
    # to 1 and 2 if a lane is dropped, so the merge must not care.
    revised = _plan(
        *[_firm_query(i, f"Firm {i}") for i in range(1, 6)],
        _archetype_query(1, "a01"),
        _archetype_query(2, "a02"),
    )
    _, resumed = bridge.start_or_resume_run(
        resume=True,
        initial_progress=revised,
        active_lanes=("people_search", "archetype"),
    )

    assert [q.id for q in resumed.queries] == [1, 2, 3, 4, 5, 6, 7]
    assert [q.archetype_id for q in resumed.queries[-2:]] == ["a01", "a02"]


def test_resume_skips_queries_whose_lane_left_the_brief(
    tmp_path: Path, capsys
) -> None:
    """A lane deleted from the brief stops spending immediately.

    Skipped, never deleted: the two team-page queries that already ran
    did real work, and their results and drops stay on the campaign's
    coverage. Only the runnable ones are stopped.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(
        _firm_query(1, "Benchmark"),
        _firm_query(2, "Benchmark", lane="team_page"),
        _firm_query(3, "Greylock", lane="team_page"),
    )
    first_run, progress = bridge.start_or_resume_run(
        resume=False, initial_progress=stored
    )
    progress.queries[1].status = "done"
    progress.queries[1].result_count = 7
    bridge.sync_progress(first_run, progress)
    capsys.readouterr()

    second_run, resumed = bridge.start_or_resume_run(
        resume=True,
        initial_progress=_plan(_firm_query(1, "Benchmark")),
        active_lanes=("people_search",),
    )

    assert _statuses(store, second_run) == {"1": "queued", "2": "done", "3": "skipped"}
    rows = {
        row["source_unit_id"]: row
        for row in store.list_work_units(second_run, kind=EXA_QUERY_KIND)
    }
    assert rows["2"]["result_count"] == 7, "a finished query keeps its work"
    assert rows["3"]["notes"] == "lane no longer in brief"
    # And the loop is given the same answer the store holds, so nothing
    # runs the skipped query on this session's pass.
    runnable = [q for q in resumed.queries if q.status in {"queued", "in_progress"}]
    assert [q.id for q in runnable] == [1]

    out = capsys.readouterr().out
    assert (
        "    [exa] plan revised — 1 query skipped (lane dropped from the brief)"
        in out
    )


def test_resume_skips_a_query_the_brief_dropped_from_a_live_lane(
    tmp_path: Path, capsys
) -> None:
    """Retiring one archetype stops that query, not the whole lane.

    The lane is still in the brief, so the lane rule says nothing — but
    the sentence someone deleted must stop searching, and the id it
    used to occupy must not be handed to whatever the compiler now
    numbers 2.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(_archetype_query(1, "a01"), _archetype_query(2, "a02"))
    first_run, _ = bridge.start_or_resume_run(resume=False, initial_progress=stored)
    capsys.readouterr()

    second_run, resumed = bridge.start_or_resume_run(
        resume=True,
        initial_progress=_plan(_archetype_query(1, "a02")),
        active_lanes=("archetype",),
    )

    assert _statuses(store, second_run) == {"1": "skipped", "2": "queued"}
    kept = next(q for q in resumed.queries if q.status == "queued")
    assert kept.archetype_id == "a02" and kept.id == 2
    assert "no longer in the brief's plan" in capsys.readouterr().out


def test_resume_requeues_skipped_queries_when_lane_returns(tmp_path: Path) -> None:
    """A lane switched back on picks up exactly what it left behind.

    Never deleting is what makes this possible: the queries are still
    there with their ids and their counters, so turning the lane back
    on costs nothing and re-runs nothing that already finished.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(
        _firm_query(1, "Benchmark"),
        _firm_query(2, "Benchmark", lane="team_page"),
    )
    first_run, _ = bridge.start_or_resume_run(resume=False, initial_progress=stored)

    both = _plan(_firm_query(1, "Benchmark"), _firm_query(2, "Benchmark", lane="team_page"))
    dropped_run, _ = bridge.start_or_resume_run(
        resume=True,
        initial_progress=_plan(_firm_query(1, "Benchmark")),
        active_lanes=("people_search",),
    )
    assert _statuses(store, dropped_run)["2"] == "skipped"

    restored_run, resumed = bridge.start_or_resume_run(
        resume=True,
        initial_progress=both,
        active_lanes=("people_search", "team_page"),
    )

    assert _statuses(store, restored_run) == {"1": "queued", "2": "queued"}
    requeued = next(q for q in resumed.queries if q.id == 2)
    assert requeued.notes == "", "the marker is lifted with the status"
    assert requeued.query == "Senior Associate at Benchmark"


def test_resume_matches_by_coverage_key_not_id(tmp_path: Path) -> None:
    """A compiled plan that renumbers everything changes nothing.

    Dropping a lane renumbers the compiler's output: the archetype that
    was id 3 becomes id 2. Matching on ids would then see "id 2
    changed" and "id 3 vanished" — re-running a paid query and
    retiring a live one on the same pass. Identity is the coverage key
    and the lane.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(
        _firm_query(1, "Benchmark"),
        _firm_query(2, "Benchmark", lane="team_page"),
        _archetype_query(3, "a01"),
    )
    first_run, progress = bridge.start_or_resume_run(
        resume=False, initial_progress=stored
    )
    # The two firm queries have run; the archetype has not, so a
    # mismatch would both retire it AND append a duplicate.
    for query in progress.queries[:2]:
        query.status = "done"
        query.result_count = 5
    bridge.sync_progress(first_run, progress)

    # The team-page lane goes; the compiler renumbers the archetype
    # from 3 to 2.
    renumbered = _plan(_firm_query(1, "Benchmark"), _archetype_query(2, "a01"))
    second_run, resumed = bridge.start_or_resume_run(
        resume=True,
        initial_progress=renumbered,
        active_lanes=("people_search", "archetype"),
    )

    assert _statuses(store, second_run) == {"1": "done", "2": "done", "3": "queued"}
    archetype = next(q for q in resumed.queries if q.lane == "archetype")
    assert archetype.id == 3, "the archetype kept the id it was stored under"
    assert len(resumed.queries) == 3, "nothing was appended as new"


def test_resume_with_unchanged_brief_changes_nothing(
    tmp_path: Path, capsys
) -> None:
    """The merge is silent and inert when the brief did not move.

    Byte-identical progress and no console line: the plan-revised lines
    are an EVENT, and a line printed on every ordinary resume is a line
    an operator learns to read past.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(_firm_query(1, "Benchmark"), _archetype_query(2, "a01"))
    first_run, progress = bridge.start_or_resume_run(
        resume=False, initial_progress=stored
    )
    progress.queries[0].status = "done"
    progress.queries[0].result_count = 9
    bridge.sync_progress(first_run, progress)

    unmerged_run, unmerged = bridge.start_or_resume_run(resume=True)
    baseline = unmerged.to_json()
    ended_at = {
        row["source_unit_id"]: row["ended_at"]
        for row in store.list_work_units(unmerged_run, kind=EXA_QUERY_KIND)
    }
    capsys.readouterr()

    merged_run, merged = bridge.start_or_resume_run(
        resume=True,
        initial_progress=_plan(_firm_query(1, "Benchmark"), _archetype_query(2, "a01")),
        active_lanes=("people_search", "archetype"),
    )

    assert merged.to_json() == baseline
    assert "plan revised" not in capsys.readouterr().out
    # No console line AND no write: an unchanged resume must not even
    # touch the rows it cloned.
    assert {
        row["source_unit_id"]: row["ended_at"]
        for row in store.list_work_units(merged_run, kind=EXA_QUERY_KIND)
    } == ended_at


def test_merge_preserves_the_cloned_drop_metrics(tmp_path: Path) -> None:
    """A brief edit must not erase what earlier sessions measured.

    ``upsert_work_unit`` OVERWRITES ``metrics_json``, so a merge that
    synced without carrying the cloned drops forward would reset every
    coverage note to "no drops" the first time anyone edited the brief
    — and the slate would read as a clean sweep of a campaign that had
    been dropping namesake results all along.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(_firm_query(1, "Benchmark"))
    first_run, progress = bridge.start_or_resume_run(
        resume=False, initial_progress=stored
    )
    progress.queries[0].status = "done"
    bridge.sync_progress(first_run, progress, metrics_by_query={"1": {"namesake": 4}})

    second_run, _ = bridge.start_or_resume_run(
        resume=True,
        initial_progress=_plan(_firm_query(1, "Benchmark"), _archetype_query(2, "a01")),
        active_lanes=("people_search", "archetype"),
    )

    assert bridge.load_query_drops(second_run) == {"1": {"namesake": 4}}


def test_payload_less_unit_is_rebuilt_from_plan_or_skipped(
    tmp_path: Path, capsys
) -> None:
    """An unreadable payload never becomes an unfiltered Exa search.

    The old reconstruction failed OPEN: it produced a query with an
    empty search string, no ``include_domains`` and no ``num_results``,
    and a runnable row like that would have been dispatched — the most
    expensive request the module can make, returning the least relevant
    people. Rebuilt from the plan where the plan still has it; skipped
    where it does not.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(_archetype_query(1, "a01"), _archetype_query(2, "a02"))
    first_run, _ = bridge.start_or_resume_run(resume=False, initial_progress=stored)
    with store.connect() as conn:
        conn.execute(
            "UPDATE work_units SET payload_json = ? WHERE run_id = ?",
            ("not json at all", first_run),
        )
        conn.commit()
    capsys.readouterr()

    second_run, resumed = bridge.start_or_resume_run(
        resume=True,
        initial_progress=_plan(_archetype_query(1, "a01")),
        active_lanes=("archetype",),
    )

    rebuilt = next(q for q in resumed.queries if q.archetype_id == "a01")
    assert rebuilt.query == "Bay Area people matching a01."
    assert rebuilt.include_domains == ["linkedin.com"]
    assert rebuilt.num_results == 25
    assert rebuilt.status == "queued"

    lost = next(q for q in resumed.queries if q.id == 2)
    assert lost.status == "skipped"
    assert lost.notes == "payload lost; not in current plan"
    assert _statuses(store, second_run) == {"1": "queued", "2": "skipped"}

    out = capsys.readouterr().out
    assert "rebuilt from the brief" in out
    assert "payload lost and not in the current plan" in out
    for line in out.splitlines():
        assert not line.lstrip().startswith("{")


def test_payload_less_done_unit_keeps_its_verdict(tmp_path: Path) -> None:
    """A finished query with a lost payload is not re-classified.

    It already ran. Rewriting it to ``skipped`` would delete a
    session's work from the campaign's coverage — the results and the
    candidates it found are in the row, and the row is the record.
    """

    store, bridge = _bridge(tmp_path)
    stored = _plan(_archetype_query(1, "a01"))
    first_run, progress = bridge.start_or_resume_run(
        resume=False, initial_progress=stored
    )
    progress.queries[0].status = "done"
    progress.queries[0].result_count = 25
    bridge.sync_progress(first_run, progress)
    with store.connect() as conn:
        conn.execute(
            "UPDATE work_units SET payload_json = ? WHERE run_id = ?",
            ("{{{", first_run),
        )
        conn.commit()

    second_run, resumed = bridge.start_or_resume_run(
        resume=True, initial_progress=_plan(), active_lanes=("people_search",)
    )

    assert _statuses(store, second_run) == {"1": "done"}
    assert resumed.queries[0].result_count == 25
