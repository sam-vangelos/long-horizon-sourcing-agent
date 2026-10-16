"""Exa source module S5 — the pipeline, against a real store.

Offline end to end: a fake Exa client serving fixture results, a fake
extractor, a fake judge returning scripted decisions, and a REAL
:class:`~shared.runtime_state.store.RuntimeStateStore` in ``tmp_path``.
Nothing here mocks the store, the lifecycle guards, the side-effect
table, the advisory lock, or the workspace destination — those are
exactly the seams the slice is about, and a mocked seam asserts the
test's belief about the seam rather than the seam.

Two disciplines run through the file:

* **assert on STORE ROWS.** ``stats`` is a session's own bookkeeping and
  would agree with a bug that never wrote anything. Candidate lifecycle
  states, work-unit statuses, side-effect rows, and ``workspace_entries``
  are what the next session actually reads, so they are what gets
  asserted.
* **the fake client charges the real meter.** Cost assertions therefore
  run through the same reserve/settle path a live run does, and the
  session-vs-campaign split is measured, not stipulated.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import exa.orchestrator as orchestrator_module
from exa.client import ExaAuthError
from exa.cost_meter import ExaBudgetExhaustedError, ExaCostMeter
from exa.judge import JudgmentOutcome, classify_verdict
from exa.orchestrator import (
    SLATE_FILENAME,
    WORKSPACE_EFFECT_TYPE,
    ExaBriefBlockedError,
    ExaPipeline,
    ExaStateDirBusyError,
)
from shared.llm_usage import record_llm_usage
from shared.runtime_state.lock import RuntimeStateLock
from shared.runtime_state.store import EXA_QUERY_KIND, RuntimeStateStore
from shared.save_destination import SaveResult
from shared.schemas import OpusDecision


# ---------------------------------------------------------------------------
# Fixtures — brief, fake transport, fake judge
# ---------------------------------------------------------------------------

_TWO_FIRM_LIST = [
    {"name": "Benchmark", "domain": "benchmark.com"},
    {"name": "Greylock", "domain": "greylock.com"},
]
_ONE_FIRM_LIST = [{"name": "Benchmark", "domain": "benchmark.com"}]


def _brief_dict(**overrides) -> dict:
    raw = {
        "role_title": "Chief of Staff to the CEO",
        "role_level": "L6",
        "role_summary": "Runs the founder's operating cadence.",
        "geography": "SF Bay Area",
        "linkedin_project": "Test Project",
        "capability_areas": [
            {"name": "Operating leverage", "description": "Ran the cadence."}
        ],
        "depth_distinction": {
            "builder_definition": "Ran the cadence.",
            "user_definition": "Attended the cadence.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "non_fit_patterns": [],
        "employer_signal_rules": [],
        "engagement_context": {"selectivity_posture": "selective"},
        "exa_query_lanes": ["people_search"],
        "exa_target_firms": list(_TWO_FIRM_LIST),
        "exa_title_terms": ["Senior Associate", "Principal"],
    }
    raw.update(overrides)
    return raw


def _write_brief(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_brief_dict(**overrides)))
    return path


def _state_dir(tmp_path: Path) -> Path:
    """A state dir that LOOKS like a real one.

    ``shared/output_paths.output_root_for_path`` walks up to the nearest
    ``output`` component, so a snapshot taken from here lands in this
    test's own ``output/runs/`` rather than the session-wide one.
    """

    path = tmp_path / "output" / "state" / "exa" / "cos"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class FakeResult:
    url: str
    title: str = ""
    text: str = ""
    highlights: list[str] = field(default_factory=list)
    published_date: str = ""

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "highlights": list(self.highlights),
            "published_date": self.published_date,
        }


@dataclass
class FakeResponse:
    results: list[FakeResult] = field(default_factory=list)
    cost_dollars: dict | None = None
    request_id: str = "fake"


def _person(handle: str, name: str, title: str, firm: str) -> FakeResult:
    return FakeResult(
        url=f"https://www.linkedin.com/in/{handle}",
        title=f"{name} - {title} - {firm} | LinkedIn",
        text=f"San Francisco, California\n{name} works at {firm}.",
        highlights=[f"{name} is a {title} at {firm}."],
    )


class FakeExaClient:
    """Serves fixture results keyed by a substring of the query text.

    Charges the meter it was built with — the same reserve/settle pair
    ``exa/client.py`` runs — so the run's Exa spend is a real number
    produced by the real meter.
    """

    USD_PER_SEARCH = 0.02

    def __init__(
        self,
        meter: ExaCostMeter,
        *,
        results: dict[str, list[FakeResult]] | None = None,
        raise_at_call: int | None = None,
        error: Exception | None = None,
    ):
        self.meter = meter
        self.results = results or {}
        self.raise_at_call = raise_at_call
        self.error = error
        self.queries: list[str] = []

    async def search(self, query: str, **kwargs) -> FakeResponse:
        call_index = len(self.queries)
        if self.raise_at_call is not None and call_index == self.raise_at_call:
            # Same ordering the real client uses: the raise happens
            # BEFORE the request is dispatched or recorded.
            raise self.error or ExaBudgetExhaustedError("cap reached")
        self.queries.append(query)
        self.meter.reserve(self.USD_PER_SEARCH)
        self.meter.settle(
            endpoint="/search",
            estimate_usd=self.USD_PER_SEARCH,
            cost_dollars={"total": self.USD_PER_SEARCH},
            request_id="fake",
            num_results=0,
        )
        for key, rows in self.results.items():
            if key in query:
                return FakeResponse(results=list(rows))
        return FakeResponse(results=[])

    def status_line(self) -> str:
        return self.meter.status_line()


def _fake_extractor(evidence, *, firm_hint="", usage_context=None):
    """No team-page lane in these briefs; a call here is a bug."""

    raise AssertionError("extractor called on a people_search-only brief")


def _decision(name: str, verdict: str, *, confidence: float = 0.8) -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision=verdict,
        path="DIRECT:1. Operating leverage",
        confidence=confidence,
        rationale=f"{name}: scripted rationale.",
        candidate_name=name,
        profile_url="",
    )


class ScriptedJudge:
    """Returns a scripted verdict per name and records every call.

    Writes a REAL usage row through ``shared.llm_usage.record_llm_usage``
    on every call — the same writer the Anthropic clients use — so the
    run's LLM spend is produced by the production cost path rather than
    by a hand-written fixture line.

    ``budget_after`` makes the judge raise the way a real budget
    exhaustion does: after N successful judgments, every further call
    raises ``ExaBudgetExhaustedError``. That is the shape the durability
    tests need — an interruption in the MIDDLE of a persisted batch.

    ``repeats`` is how many times one person may be judged: 1 everywhere
    but the re-judge tests, whose slate verdicts are judged on two samples.
    ``sequences`` scripts those samples per name, one verdict per call.
    Calls can arrive on the re-judge pass's worker threads, so the
    bookkeeping holds a lock.
    """

    # 1000 in / 200 out on the Opus rate = $0.01 exactly.
    USD_PER_CALL = 0.01

    def __init__(
        self,
        verdicts: dict[str, str],
        default: str = "REJECT",
        *,
        budget_after: int | None = None,
        repeats: int = 1,
        sequences: dict[str, list[str]] | None = None,
    ):
        self.verdicts = verdicts
        self.default = default
        self.budget_after = budget_after
        self.repeats = repeats
        self.sequences = {name: list(seq) for name, seq in (sequences or {}).items()}
        self.seen: list[str] = []
        self.identity_keys: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, candidate, brief, *, query=None) -> JudgmentOutcome:
        with self._lock:
            if self.budget_after is not None and len(self.seen) >= self.budget_after:
                raise ExaBudgetExhaustedError("judge budget reached")
            self.seen.append(candidate.name)
            assert self.identity_keys.count(candidate.identity_key) < self.repeats, (
                f"{candidate.identity_key} judged more than {self.repeats} time(s)"
            )
            self.identity_keys.append(candidate.identity_key)
            sequence = self.sequences.get(candidate.name)
            verdict = (
                sequence.pop(0)
                if sequence
                else self.verdicts.get(candidate.name, self.default)
            )
        record_llm_usage(
            provider="anthropic",
            model="claude-opus-4-1-20250805",
            usage={"input_tokens": 1000, "output_tokens": 200},
            request={"stage": "full"},
        )
        decision = _decision(candidate.name, verdict)
        # The PRODUCTION classifier, not a hand-written map: a fake that
        # buckets verdicts by its own rules would hide a change in
        # ``exa.judge.classify_verdict`` from every test in this file.
        return JudgmentOutcome(full=decision, bucket=classify_verdict(decision))


def _pipeline(
    tmp_path: Path,
    *,
    brief_path: Path | None = None,
    results: dict[str, list[FakeResult]] | None = None,
    judge=None,
    raise_at_call: int | None = None,
    error: Exception | None = None,
    known_from=(),
    **pipeline_kwargs,
) -> tuple[ExaPipeline, list[FakeExaClient]]:
    clients: list[FakeExaClient] = []

    def factory(meter: ExaCostMeter) -> FakeExaClient:
        client = FakeExaClient(
            meter,
            results=results,
            raise_at_call=raise_at_call,
            error=error,
        )
        clients.append(client)
        return client

    pipeline = ExaPipeline(
        brief_path=brief_path or _write_brief(tmp_path),
        output_dir=_state_dir(tmp_path),
        client_factory=factory,
        judge=judge or ScriptedJudge({}),
        extractor=_fake_extractor,
        known_from=known_from,
        **pipeline_kwargs,
    )
    return pipeline, clients


def _store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(_state_dir(tmp_path) / "runtime_state.sqlite3")


def _rows(tmp_path: Path, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect(_state_dir(tmp_path) / "runtime_state.sqlite3")
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(sql, params).fetchall())
    finally:
        conn.close()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


_TWO_FIRMS = {
    "Benchmark": [_person("averylinholt", "Avery Holt", "Investor", "Benchmark")],
    "Greylock": [_person("mlorne", "Mira Lorne", "Principal", "Greylock")],
}

_THREE_AT_BENCHMARK = {
    "Benchmark": [
        _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
        _person("theomarsh", "Theo Marsh", "Principal", "Benchmark"),
        _person("quinnxavila", "Quinn Avila", "Investor", "Benchmark"),
    ]
}


# ---------------------------------------------------------------------------
# Lint gate and the single-writer lock
# ---------------------------------------------------------------------------


def test_blocking_lint_stops_before_any_spend_or_state(tmp_path: Path) -> None:
    """CLO-115: a blacklisted target firm is an ERROR, and it fires first."""

    brief_path = _write_brief(tmp_path, employer_blacklist=["Benchmark"])
    pipeline, clients = _pipeline(tmp_path, brief_path=brief_path)

    with pytest.raises(ExaBriefBlockedError):
        asyncio.run(pipeline.run())

    assert clients == [], "no Exa client was constructed, so nothing was spent"
    state_dir = _state_dir(tmp_path)
    assert not (state_dir / "runtime_state.sqlite3").exists()
    assert not (state_dir / "runtime_state.lock").exists()
    assert not (state_dir / "progress.json").exists()


def test_second_process_on_same_state_dir_is_refused(tmp_path: Path, capsys) -> None:
    """One writer per state dir, refused before reconcile or clone.

    Two orchestrators on one campaign is not a slow run, it is a
    corrupted one: both would reconcile the other's open attempts, both
    would clone work units into rival run rows, and both would judge the
    same people. The advisory lock is taken before the store is opened,
    so the loser writes nothing at all.
    """

    state_dir = _state_dir(tmp_path)
    holder = RuntimeStateLock(state_dir)
    holder.acquire()
    try:
        pipeline, clients = _pipeline(tmp_path, results=_TWO_FIRMS)
        with pytest.raises(ExaStateDirBusyError):
            asyncio.run(pipeline.run())
    finally:
        holder.release()

    assert clients == []
    assert not (state_dir / "runtime_state.sqlite3").exists(), (
        "the loser opened no store, so it created no run row"
    )
    out = capsys.readouterr().out
    assert "    [exa] NO-GO — another Exa process holds this state dir" in out
    assert len([line for line in out.splitlines() if line.strip()]) == 1


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_run_judges_each_kept_candidate_once(tmp_path: Path) -> None:
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "REJECT"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Avery Holt", "Mira Lorne"]
    assert stats.judged == 2
    assert stats.saved == 1
    assert stats.rejected == 1
    assert stats.queries_run == 2

    store = _store(tmp_path)
    rows = store.list_work_units(stats.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in rows] == ["done", "done"]
    assert store.get_run(stats.run_id)["status"] == "completed"

    saved = store.get_candidate(
        source="exa",
        brief_id=pipeline.brief_id,
        identity_key="linkedin.com/in/averylinholt",
    )
    assert saved["current_lifecycle_state"] == "full_terminal"
    assert saved["terminal_decision"] == "SAVE"


def test_do_not_save_person_is_never_judged_or_saved(tmp_path: Path) -> None:
    brief_path = _write_brief(
        tmp_path,
        do_not_save=[{"name": "Avery Holt", "reason": "client conflict"}],
    )
    judge = ScriptedJudge({"Mira Lorne": "SAVE"})
    pipeline, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=_TWO_FIRMS, judge=judge
    )

    stats = asyncio.run(pipeline.run())

    assert "Avery Holt" not in judge.seen
    assert judge.seen == ["Mira Lorne"]
    assert stats.skipped.get("do_not_save") == 1

    skipped = _jsonl(_state_dir(tmp_path) / "skipped.jsonl")
    assert any(
        row["name"] == "Avery Holt" and row["reason"].startswith("do_not_save:")
        for row in skipped
    )
    store = _store(tmp_path)
    assert (
        store.get_candidate(
            source="exa",
            brief_id=pipeline.brief_id,
            identity_key="linkedin.com/in/averylinholt",
        )
        is None
    ), "a do-not-save person never reaches a candidate row"


def test_save_bucket_writes_workspace_entry_and_saves_jsonl(
    tmp_path: Path,
) -> None:
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "REJECT"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    stats = asyncio.run(pipeline.run())

    entries = _rows(tmp_path, "SELECT * FROM workspace_entries")
    effects = _rows(tmp_path, "SELECT * FROM side_effects")

    assert len(entries) == 1
    assert entries[0]["save_decision"] == "SAVE"
    assert entries[0]["surface_type"] == "exa_web_evidence"
    context = json.loads(entries[0]["contextualization_payload_json"])
    assert context["verify_before_outreach"] is True
    assert context["evidence_urls"]

    assert len(effects) == 1
    assert effects[0]["effect_type"] == WORKSPACE_EFFECT_TYPE
    assert effects[0]["status"] == "succeeded"
    assert effects[0]["idempotency_key"] == (
        f"exa:{pipeline.brief_id}:linkedin.com/in/averylinholt:workspace"
    )

    saves = _jsonl(_state_dir(tmp_path) / "saves.jsonl")
    assert [row["name"] for row in saves] == ["Avery Holt"]
    assert saves[0]["run_id"] == stats.run_id
    assert saves[0]["verify_before_outreach"] is True


def test_failed_workspace_write_is_not_counted_as_a_save(tmp_path: Path) -> None:
    """"The judge said SAVE" is not "the save landed" (LinkedIn's P1.2)."""

    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "REJECT"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    from shared.save_destination.candidate_workspace import (
        CandidateWorkspaceSaveDestination,
    )

    original = CandidateWorkspaceSaveDestination.save
    try:
        CandidateWorkspaceSaveDestination.save = lambda self, **kw: SaveResult(
            destination="candidate_workspace", status="failed", error="disk on fire"
        )
        stats = asyncio.run(pipeline.run())
    finally:
        CandidateWorkspaceSaveDestination.save = original

    assert stats.saved == 0, "a save that did not land is not a save"
    assert not (_state_dir(tmp_path) / "saves.jsonl").exists()
    effects = _rows(tmp_path, "SELECT * FROM side_effects")
    assert [row["status"] for row in effects] == ["failed"]
    # The verdict itself is untouched — the judge's answer is still on
    # the record, ready for the next session's replay pass.
    store = _store(tmp_path)
    row = store.get_candidate(
        source="exa",
        brief_id=pipeline.brief_id,
        identity_key="linkedin.com/in/averylinholt",
    )
    assert row["terminal_decision"] == "SAVE"


def test_review_bucket_is_recorded_not_saved(tmp_path: Path) -> None:
    judge = ScriptedJudge(
        {"Avery Holt": "REVIEW_FLAGGED", "Mira Lorne": "REVIEW_INFERRED"}
    )
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert stats.review == 2
    assert stats.saved == 0
    assert not (_state_dir(tmp_path) / "saves.jsonl").exists()
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []

    store = _store(tmp_path)
    row = store.get_candidate(
        source="exa",
        brief_id=pipeline.brief_id,
        identity_key="linkedin.com/in/averylinholt",
    )
    assert row["terminal_decision"] == "REVIEW_FLAGGED"
    assert row["current_lifecycle_state"] == "full_terminal"


def test_failure_bucket_is_retryable_not_a_rejection(tmp_path: Path) -> None:
    """A judgment that did not happen is not a verdict about the person.

    ``JUDGMENT_FAILURE`` lands the candidate in ``failed_retryable``, the
    one non-terminal failure state — so the next session re-judges them
    instead of inheriting a rejection the model never made.
    """

    judge = ScriptedJudge({"Avery Holt": "JUDGMENT_FAILURE"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert stats.failed == 1
    assert stats.rejected == 1  # Mira Lorne, a real REJECT
    assert stats.saved == 0

    store = _store(tmp_path)
    row = store.get_candidate(
        source="exa",
        brief_id=pipeline.brief_id,
        identity_key="linkedin.com/in/averylinholt",
    )
    assert row["current_lifecycle_state"] == "failed_retryable"
    assert row["terminal_decision"] is None
    assert "linkedin.com/in/averylinholt" not in store.list_terminal_identity_keys(
        source="exa", brief_id=pipeline.brief_id
    ), "a failed judgment must not suppress the person on resume"


# ---------------------------------------------------------------------------
# Durability — the batch is canonical before the first Opus call
# ---------------------------------------------------------------------------


def test_batch_is_persisted_before_first_judgment(tmp_path: Path) -> None:
    """Exa spend becomes durable state before any model spend happens.

    The judge raises on candidate ONE of three. All three must already be
    in the store: the campaign paid Exa for that page of results, and an
    interruption during judgment must never cost it the harvest.
    """

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    judge = ScriptedJudge({}, budget_after=0)
    pipeline, _ = _pipeline(
        tmp_path, brief_path=brief, results=_THREE_AT_BENCHMARK, judge=judge
    )

    stats = asyncio.run(pipeline.run())

    assert stats.paused is True
    assert judge.seen == [], "the budget raised before the first verdict"

    rows = _rows(
        tmp_path,
        "SELECT identity_key, person_key, current_lifecycle_state FROM candidates "
        "ORDER BY identity_key",
    )
    assert [row["identity_key"] for row in rows] == [
        "linkedin.com/in/averylinholt",
        "linkedin.com/in/quinnxavila",
        "linkedin.com/in/theomarsh",
    ]
    for row in rows:
        assert row["person_key"], "person_key is written at persist time"
    # The one the judge reached is retryable; the rest never started.
    states = {row["identity_key"]: row["current_lifecycle_state"] for row in rows}
    assert states["linkedin.com/in/averylinholt"] == "failed_retryable"
    assert states["linkedin.com/in/theomarsh"] == "snippet_extracted"
    assert states["linkedin.com/in/quinnxavila"] == "snippet_extracted"

    # The QUERY is done, not requeued: its Exa half completed and its
    # batch is durable. Re-running it would re-spend Exa for nothing.
    work_units = _store(tmp_path).list_work_units(stats.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in work_units] == ["done"]

    # And every record is retrievable, so the next session needs no Exa.
    payloads = _rows(
        tmp_path,
        "SELECT payload_json FROM candidate_attempts WHERE stage = 'snippet'",
    )
    assert len(payloads) == 3
    for row in payloads:
        record = json.loads(row["payload_json"])["candidate_record"]
        assert record["name"] and record["evidence"]


def test_resume_rejudges_pending_candidates_without_exa_spend(
    tmp_path: Path,
) -> None:
    """The second session judges the leftovers and makes ZERO Exa calls.

    The brief compiles one query, and that query is already ``done``, so
    the client is never even constructed — the strongest available form
    of "no Exa call was possible".
    """

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    judge_one = ScriptedJudge({"Avery Holt": "SAVE"}, budget_after=1)
    first, clients_one = _pipeline(
        tmp_path, brief_path=brief, results=_THREE_AT_BENCHMARK, judge=judge_one
    )
    stats_one = asyncio.run(first.run())

    assert judge_one.seen == ["Avery Holt"]
    assert stats_one.paused is True
    assert clients_one[0].queries != []

    judge_two = ScriptedJudge(
        {"Theo Marsh": "SAVE", "Quinn Avila": "REVIEW_FLAGGED"}
    )
    second, clients_two = _pipeline(
        tmp_path, brief_path=brief, results=_THREE_AT_BENCHMARK, judge=judge_two
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert clients_two == [], "no Exa client was constructed on the resume"
    assert sorted(judge_two.seen) == ["Quinn Avila", "Theo Marsh"], (
        "the two the budget never reached are judged; Avery was already "
        "terminal in session one and is not re-judged"
    )
    assert stats_two.judged == 2
    assert stats_two.replayed_saves == 0, (
        "Avery's save landed in session one, so there is nothing to replay"
    )

    states = {
        row["identity_key"]: (
            row["current_lifecycle_state"],
            row["terminal_decision"],
        )
        for row in _rows(
            tmp_path,
            "SELECT identity_key, current_lifecycle_state, terminal_decision "
            "FROM candidates",
        )
    }
    assert states["linkedin.com/in/averylinholt"] == ("full_terminal", "SAVE")
    assert states["linkedin.com/in/theomarsh"] == ("full_terminal", "SAVE")
    assert states["linkedin.com/in/quinnxavila"] == (
        "full_terminal",
        "REVIEW_FLAGGED",
    )
    assert _store(tmp_path).get_run(stats_two.run_id)["status"] == "completed"
    # Two saves across two sessions, one workspace row each.
    assert len(_rows(tmp_path, "SELECT * FROM workspace_entries")) == 2
    assert len(_jsonl(_state_dir(tmp_path) / "saves.jsonl")) == 2


def test_retryable_failure_is_rejudged_on_resume(tmp_path: Path) -> None:
    """A parse failure is a retry, and the retry actually happens."""

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    judge_one = ScriptedJudge({"Avery Holt": "JUDGMENT_FAILURE"})
    first, _ = _pipeline(
        tmp_path,
        brief_path=brief,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=judge_one,
    )
    asyncio.run(first.run())

    assert _rows(
        tmp_path, "SELECT current_lifecycle_state FROM candidates"
    )[0]["current_lifecycle_state"] == "failed_retryable"

    judge_two = ScriptedJudge({"Avery Holt": "SAVE"})
    second, clients_two = _pipeline(
        tmp_path,
        brief_path=brief,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=judge_two,
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert clients_two == [], "the re-judge costs no Exa call"
    assert judge_two.seen == ["Avery Holt"]
    row = _rows(
        tmp_path,
        "SELECT current_lifecycle_state, terminal_decision FROM candidates",
    )[0]
    assert row["current_lifecycle_state"] == "full_terminal"
    assert row["terminal_decision"] == "SAVE"
    assert stats_two.saved == 1
    assert _rows(tmp_path, "SELECT * FROM workspace_entries")[0]["save_decision"] == (
        "SAVE"
    )


def test_interrupted_save_is_replayed_on_resume(tmp_path: Path) -> None:
    """A verdict without its side effect is finished, exactly once.

    Session one dies BETWEEN the terminal SAVE and the workspace write,
    leaving a ``pending`` side-effect row. Nothing would ever revisit
    that person on its own — a SAVE-family terminal suppresses
    re-judgement by design — so the replay pass is the only thing
    standing between "the campaign thinks it saved her" and a workspace
    that has never heard of her.
    """

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    judge_one = ScriptedJudge({"Avery Holt": "SAVE"})
    first, _ = _pipeline(
        tmp_path,
        brief_path=brief,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=judge_one,
    )

    from shared.save_destination.candidate_workspace import (
        CandidateWorkspaceSaveDestination,
    )

    class _Killed(RuntimeError):
        pass

    original = CandidateWorkspaceSaveDestination.save
    try:
        def _die(self, **kwargs):
            raise _Killed("process killed mid-save")

        CandidateWorkspaceSaveDestination.save = _die
        with pytest.raises(_Killed):
            asyncio.run(first.run())
    finally:
        CandidateWorkspaceSaveDestination.save = original

    # The verdict landed; the side effect is stranded as pending.
    row = _rows(
        tmp_path,
        "SELECT current_lifecycle_state, terminal_decision FROM candidates",
    )[0]
    assert row["current_lifecycle_state"] == "full_terminal"
    assert row["terminal_decision"] == "SAVE"
    effects = _rows(tmp_path, "SELECT status FROM side_effects")
    assert [effect["status"] for effect in effects] == ["pending"]
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []
    assert not (_state_dir(tmp_path) / "saves.jsonl").exists()

    judge_two = ScriptedJudge({})
    second, clients_two = _pipeline(
        tmp_path,
        brief_path=brief,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=judge_two,
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert judge_two.seen == [], "the replay re-judges nobody"
    assert clients_two == [], "and costs no Exa call"
    assert stats_two.replayed_saves == 1
    assert stats_two.saved == 1

    entries = _rows(tmp_path, "SELECT * FROM workspace_entries")
    assert len(entries) == 1
    assert entries[0]["save_decision"] == "SAVE"
    effects = _rows(tmp_path, "SELECT status, attempt_count FROM side_effects")
    assert [effect["status"] for effect in effects] == ["succeeded"]
    saves = _jsonl(_state_dir(tmp_path) / "saves.jsonl")
    assert len(saves) == 1, "saves.jsonl is written once, on success"

    # A THIRD session must not replay it again.
    third, _ = _pipeline(
        tmp_path,
        brief_path=brief,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=ScriptedJudge({}),
    )
    stats_three = asyncio.run(third.run(resume=True))
    assert stats_three.replayed_saves == 0
    assert len(_jsonl(_state_dir(tmp_path) / "saves.jsonl")) == 1
    assert len(_rows(tmp_path, "SELECT * FROM workspace_entries")) == 1


def test_resume_never_writes_a_review_bucket_save_to_the_workspace(
    tmp_path: Path,
) -> None:
    """A hedged save is a lead to look at, on resume as on the live path.

    TRANSFERABLE_SAVE and SIGNAL_SAVE are in the save family but classify to
    review, so the live path never writes them to the workspace. Their
    side-effect row is therefore always absent, and the replay pass must not
    mistake that for an interrupted save.
    """

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    judge_one = ScriptedJudge(
        {"Avery Holt": "TRANSFERABLE_SAVE", "Theo Marsh": "SIGNAL_SAVE"}
    )
    first, _ = _pipeline(
        tmp_path, brief_path=brief, results=_THREE_AT_BENCHMARK, judge=judge_one
    )
    stats_one = asyncio.run(first.run())
    assert stats_one.review == 2 and stats_one.saved == 0
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []

    second, _ = _pipeline(
        tmp_path,
        brief_path=brief,
        results=_THREE_AT_BENCHMARK,
        judge=ScriptedJudge({}),
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert stats_two.replayed_saves == 0
    assert stats_two.saved == 0
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []
    assert not (_state_dir(tmp_path) / "saves.jsonl").exists()


# ---------------------------------------------------------------------------
# Resume across queries
# ---------------------------------------------------------------------------


def test_resume_skips_done_queries_and_seen_people(tmp_path: Path) -> None:
    """The second session re-runs only what is left, and re-judges nobody.

    ``ScriptedJudge`` asserts on a repeated identity key, so a person
    re-judged across the two sessions fails the test from inside the
    judge rather than in a counter at the end.
    """

    brief_path = _write_brief(tmp_path)
    judge = ScriptedJudge({"Avery Holt": "SAVE"})

    # Session one stops after the first query.
    first, clients_one = _pipeline(
        tmp_path, brief_path=brief_path, results=_TWO_FIRMS, judge=judge
    )
    stats_one = asyncio.run(first.run(max_queries=1))
    assert judge.seen == ["Avery Holt"]
    assert clients_one[0].queries == [
        "Senior Associate or Principal at Benchmark SF Bay Area"
    ]
    # An operator-imposed stop with work left is INTERRUPTED, not a
    # completed campaign.
    store = _store(tmp_path)
    first_run = store.get_run(stats_one.run_id)
    assert first_run["status"] == "interrupted"
    assert first_run["stop_reason"] == "operator_stop"
    assert stats_one.paused is True

    # Session two resumes. Query 1 is done; query 2 runs — and its
    # fixture also returns Avery Holt, who must not be re-judged.
    both_firms = {
        "Benchmark": _TWO_FIRMS["Benchmark"],
        "Greylock": [
            _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
            _person("mlorne", "Mira Lorne", "Principal", "Greylock"),
        ],
    }
    second, clients_two = _pipeline(
        tmp_path, brief_path=brief_path, results=both_firms, judge=judge
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert clients_two[0].queries == [
        "Senior Associate or Principal at Greylock SF Bay Area"
    ], "the done query was not re-run"
    assert judge.seen == ["Avery Holt", "Mira Lorne"]
    assert stats_two.skipped.get("already_seen") == 1

    rows = store.list_work_units(stats_two.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in rows] == ["done", "done"]
    assert store.get_run(stats_two.run_id)["status"] == "completed"
    assert stats_one.run_id != stats_two.run_id


def test_max_queries_with_nothing_left_still_completes(tmp_path: Path) -> None:
    """The stop only fires when work actually remains."""

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    judge = ScriptedJudge({"Avery Holt": "SAVE"})
    pipeline, _ = _pipeline(
        tmp_path,
        brief_path=brief,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=judge,
    )

    stats = asyncio.run(pipeline.run(max_queries=5))

    assert stats.paused is False
    assert _store(tmp_path).get_run(stats.run_id)["status"] == "completed"


# ---------------------------------------------------------------------------
# Budget and auth
# ---------------------------------------------------------------------------


def test_budget_exhaustion_pauses_with_query_requeued(tmp_path: Path) -> None:
    """An EXA-side raise requeues; the query never dispatched."""

    judge = ScriptedJudge({"Avery Holt": "SAVE"})
    pipeline, _ = _pipeline(
        tmp_path, results=_TWO_FIRMS, judge=judge, raise_at_call=1
    )

    stats = asyncio.run(pipeline.run())

    assert stats.paused is True
    assert stats.queries_run == 1

    store = _store(tmp_path)
    run = store.get_run(stats.run_id)
    assert run["status"] == "interrupted"
    assert run["stop_reason"] == "api_budget_exhausted"
    rows = store.list_work_units(stats.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in rows] == ["done", "queued"], (
        "the query that never dispatched goes back to queued, not in_progress"
    )
    # The close-out still ran: the slate and the cost receipt exist.
    assert (_state_dir(tmp_path) / SLATE_FILENAME).exists()
    assert (_state_dir(tmp_path) / "cost_rollup.json").exists()


def test_judge_side_budget_leaves_the_query_done(tmp_path: Path) -> None:
    """A judge-side pause must NOT requeue a query whose search completed.

    Requeuing it would re-run the Exa search on resume — paying twice for
    a page of results the campaign already holds — and the batch is
    already durable, so there is nothing to re-acquire.
    """

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    judge = ScriptedJudge({"Avery Holt": "SAVE"}, budget_after=1)
    pipeline, clients = _pipeline(
        tmp_path, brief_path=brief, results=_THREE_AT_BENCHMARK, judge=judge
    )

    stats = asyncio.run(pipeline.run())

    assert stats.paused is True
    assert len(clients[0].queries) == 1
    work_units = _store(tmp_path).list_work_units(stats.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in work_units] == ["done"]
    assert _store(tmp_path).get_run(stats.run_id)["stop_reason"] == (
        "api_budget_exhausted"
    )


def test_auth_error_finishes_run_as_error_and_propagates(
    tmp_path: Path,
) -> None:
    pipeline, _ = _pipeline(
        tmp_path,
        results=_TWO_FIRMS,
        judge=ScriptedJudge({}),
        raise_at_call=0,
        error=ExaAuthError("401 INVALID_API_KEY"),
    )

    with pytest.raises(ExaAuthError):
        asyncio.run(pipeline.run())

    store = _store(tmp_path)
    run = store.get_latest_run(source="exa", brief_id=pipeline.brief_id)
    assert run["status"] == "error"


# ---------------------------------------------------------------------------
# Cost, snapshot, console, identity
# ---------------------------------------------------------------------------


def test_pipeline_end_carries_session_and_campaign_costs(tmp_path: Path) -> None:
    """Session spend on the run-dir plane, campaign spend beside it.

    ``cost_usd`` stays session-scoped because that is what a per-run-dir
    rollup means; the campaign figures come from the append-only
    receipts, which span every session. After a resume the two must
    DISAGREE, or one of them is lying.
    """

    from shared.cost_rollup import _module_cost_from_state_dir, aggregate_cost_for_run

    brief_path = _write_brief(tmp_path)
    judge = ScriptedJudge({"Avery Holt": "SAVE"})
    first, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=_TWO_FIRMS, judge=judge
    )
    asyncio.run(first.run(max_queries=1))

    second, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=_TWO_FIRMS, judge=judge
    )
    stats = asyncio.run(second.run(resume=True))

    state_dir = _state_dir(tmp_path)
    session_exa = FakeExaClient.USD_PER_SEARCH
    session_llm = ScriptedJudge.USD_PER_CALL
    campaign_exa = 2 * FakeExaClient.USD_PER_SEARCH
    campaign_llm = 2 * ScriptedJudge.USD_PER_CALL

    rows = _jsonl(state_dir / "run_log.jsonl")
    pipeline_end = [row for row in rows if row.get("event") == "pipeline_end"]
    assert len(pipeline_end) == 2, "one per session"
    latest = pipeline_end[-1]
    assert latest["cost_usd"] == pytest.approx(session_exa + session_llm, abs=1e-6)
    assert latest["exa_cost_usd"] == pytest.approx(session_exa, abs=1e-6)
    assert latest["llm_cost_usd"] == pytest.approx(session_llm, abs=1e-6)
    assert latest["campaign_exa_cost_usd"] == pytest.approx(campaign_exa, abs=1e-6)
    assert latest["campaign_llm_cost_usd"] == pytest.approx(campaign_llm, abs=1e-6)

    assert stats.exa_cost_usd == pytest.approx(session_exa, abs=1e-6)
    assert stats.llm_cost_usd == pytest.approx(session_llm, abs=1e-6)
    assert stats.campaign_exa_cost_usd == pytest.approx(campaign_exa, abs=1e-6)
    assert stats.campaign_llm_cost_usd == pytest.approx(campaign_llm, abs=1e-6)

    # The receipts the campaign figures were read out of.
    usage_rows = _jsonl(state_dir / "token-cost-log.jsonl")
    assert len(usage_rows) == 2
    assert {row["source"] for row in usage_rows} == {"exa"}
    assert {row["brief_id"] for row in usage_rows} == {second.brief_id}
    assert len({row["run_id"] for row in usage_rows}) == 2, "one row per session"
    assert len(_jsonl(state_dir / "exa-usage.jsonl")) == 2

    # The run-dir rollup reads the SESSION number — that is the contract.
    module_cost = _module_cost_from_state_dir("exa", state_dir)
    assert module_cost is not None
    assert module_cost.cost_usd == pytest.approx(session_exa + session_llm, abs=1e-6)
    assert module_cost.sources == ["run_log.pipeline_end.cost_usd"]
    assert aggregate_cost_for_run({"exa": state_dir}).total_usd == pytest.approx(
        session_exa + session_llm, abs=1e-6
    )

    # And the slate carries both, campaign first.
    lede = (state_dir / SLATE_FILENAME).read_text().splitlines()[2]
    assert f"spend ${campaign_exa:.2f} Exa + ${campaign_llm:.2f} LLM" in lede
    assert f"${session_exa:.2f} Exa + ${session_llm:.2f} LLM" in lede
    assert lede.index("Campaign to date") < lede.index("This session")


def test_slate_is_campaign_wide_after_resume(tmp_path: Path) -> None:
    """Both sessions' people are on one slate, read from SQLite.

    Alice is judged in run 1 and Bob in run 2. A slate built from this
    session's counters would show one of them; a slate built from the
    campaign plane shows both, and the counts agree with the store.
    """

    brief_path = _write_brief(tmp_path)
    people = {
        "Benchmark": [_person("alicechen", "Alice Chen", "Principal", "Benchmark")],
        "Greylock": [_person("bobrivera", "Bob Rivera", "Investor", "Greylock")],
    }
    judge = ScriptedJudge(
        {"Alice Chen": "REVIEW_FLAGGED", "Bob Rivera": "SAVE"}
    )

    first, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=people, judge=judge
    )
    asyncio.run(first.run(max_queries=1))

    slate_after_one = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Alice Chen" in slate_after_one
    assert "Bob Rivera" not in slate_after_one

    second, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=people, judge=judge
    )
    stats_two = asyncio.run(second.run(resume=True))

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "## The short list — 1" in slate
    assert "## Worth a look — 1" in slate
    assert "Bob Rivera" in slate.split("## The short list", 1)[1].split(
        "## Worth a look", 1
    )[0]
    assert "Alice Chen" in slate.split("## Worth a look", 1)[1].split(
        "## Already known", 1
    )[0]
    # The session judged one person; the campaign judged two.
    assert stats_two.judged == 1
    lede = slate.splitlines()[2]
    assert "This session (run 2): 1 judged" in lede
    assert "2 queries returned 2 results and produced 2 candidates" in lede

    # And the counts agree with the canonical plane.
    terminal = _rows(
        tmp_path,
        "SELECT terminal_decision FROM candidates WHERE terminal_decision IS NOT NULL",
    )
    assert sorted(row["terminal_decision"] for row in terminal) == [
        "REVIEW_FLAGGED",
        "SAVE",
    ]

    # Coverage survives the resume too — the per-firm rows come from the
    # cloned work units, not from this session's memory.
    assert "**Benchmark** — 1 query, 1 result, 1 candidate" in slate
    assert "**Greylock** — 1 query, 1 result, 1 candidate" in slate


def test_coverage_lists_each_archetype_separately(tmp_path: Path) -> None:
    """Each archetype query gets its own row, with what it actually placed.

    The regression this locks: archetype queries carry no firm, so
    keying coverage on ``query.firm`` folded every one of them into a
    single "(no firm)" row. Six hypotheses about who to look for
    reported as one anonymous line, and the recruiter's next move —
    rewrite the sentence that found nobody — needs to know WHICH
    sentence that was.

    The short-list and worth-a-look columns are the load-bearing half.
    Results and candidates only say a query retrieved people; a query
    can retrieve twenty-five and place none, and on counts alone that
    reads as a healthy row.
    """

    brief_path = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        exa_query_lanes=["people_search", "archetype"],
        exa_archetype_queries=[
            {
                "id": "a01-mbb-to-vc",
                "axis": "consulting_to_investing",
                "text": (
                    "Former McKinsey consultants in the SF Bay Area who moved "
                    "into venture capital deal teams."
                ),
            },
            {
                "id": "a02-investor-to-cos",
                "text": (
                    "Former SF Bay Area investors now Chief of Staff at an AI "
                    "company."
                ),
            },
        ],
    )
    results = {
        "Benchmark": [
            _person("averylinholt", "Avery Holt", "Investor", "Benchmark")
        ],
        # a01 retrieves two people and places both.
        "moved into venture capital deal teams": [
            _person("dmarlow", "Dana Marlow", "Principal", "Sutter Hill"),
            _person("iokonkwo", "Ify Okonkwo", "Vice President", "Ridgeline"),
        ],
        # a02 retrieves one person and places none — the row that has
        # to be distinguishable from a01's.
        "now Chief of Staff at an AI company": [
            _person("ribarra", "Rafael Ibarra", "Chief of Staff", "Rowan AI")
        ],
    }
    judge = ScriptedJudge(
        {
            "Avery Holt": "SAVE",
            "Dana Marlow": "SAVE",
            "Ify Okonkwo": "REVIEW_FLAGGED",
            "Rafael Ibarra": "REJECT",
        }
    )

    pipeline, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=results, judge=judge
    )
    asyncio.run(pipeline.run())

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    coverage = slate.split("## Coverage notes", 1)[1].split(
        "## Verify before outreach", 1
    )[0]

    # The firm row is untouched, and no archetype is hiding inside it.
    assert "**Benchmark** — 1 query, 1 result, 1 candidate" in coverage
    assert "(no firm)" not in coverage

    # One row per archetype, under their own heading, in brief order.
    assert "### Archetype queries" in coverage
    archetype_block = coverage.split("### Archetype queries", 1)[1]
    assert archetype_block.index("a01-mbb-to-vc") < archetype_block.index(
        "a02-investor-to-cos"
    )
    assert (
        "- **a01-mbb-to-vc** — 2 results, 2 candidates, 1 short list, "
        "1 worth a look; no drops." in archetype_block
    )
    assert (
        "- **a02-investor-to-cos** — 1 result, 1 candidate, 0 short list, "
        "0 worth a look; no drops." in archetype_block
    )
    assert (
        "Found people, placed none — the shape as written is retrieving the "
        "wrong population." in archetype_block
    )

    # The store agrees: family keys are namespaced, so a firm and an
    # archetype can never land in the same bucket.
    family_keys = {
        row["family_key"]
        for row in _rows(tmp_path, "SELECT family_key FROM work_units")
    }
    assert family_keys == {
        "Benchmark",
        "archetype:a01-mbb-to-vc",
        "archetype:a02-investor-to-cos",
    }

    # And the structured report carries the same rows, so a consumer
    # that never reads the markdown sees the split too.
    report = json.loads((_state_dir(tmp_path) / "run-report.json").read_text())
    performance = {row["firm"]: row for row in report["string_performance"]}
    assert set(performance) == {
        "Benchmark",
        "archetype:a01-mbb-to-vc",
        "archetype:a02-investor-to-cos",
    }
    a01 = performance["archetype:a01-mbb-to-vc"]
    assert a01["archetype_id"] == "a01-mbb-to-vc"
    assert (a01["short_list"], a01["worth_a_look"]) == (1, 1)
    a02 = performance["archetype:a02-investor-to-cos"]
    assert (a02["short_list"], a02["worth_a_look"]) == (0, 0)
    # Nothing returned zero candidates, so nothing is a coverage gap.
    assert report["coverage_gaps"] == []


def test_report_failure_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A missing report is DEGRADED, not a failed campaign.

    The run status stays what the loop decided, and the failure reaches
    the operator's surface — a WARNING in a logger nothing tails is the
    same as silence.
    """

    def _boom(view):
        raise ValueError("renderer exploded")

    monkeypatch.setattr(orchestrator_module, "render_slate", _boom)

    judge = ScriptedJudge({"Avery Holt": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    stats = asyncio.run(pipeline.run())

    store = _store(tmp_path)
    assert store.get_run(stats.run_id)["status"] == "completed"
    assert not (_state_dir(tmp_path) / SLATE_FILENAME).exists()
    # The costs step ran BEFORE the report, so its artifact survived.
    assert (_state_dir(tmp_path) / "cost_rollup.json").exists()

    out = capsys.readouterr().out
    degraded = [line for line in out.splitlines() if "close-out degraded" in line]
    assert len(degraded) == 1
    assert degraded[0].startswith("    [exa] close-out degraded — report: ValueError")

    events = _rows(
        tmp_path,
        "SELECT payload_json FROM events WHERE event_type = 'close_out_degraded'",
    )
    assert len(events) == 1
    assert json.loads(events[0]["payload_json"])["step"] == "report"


def test_snapshot_finalizes_into_output_runs(tmp_path: Path) -> None:
    judge = ScriptedJudge({"Avery Holt": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    stats = asyncio.run(pipeline.run())

    runs_root = tmp_path / "output" / "runs" / "exa"
    run_dirs = [
        p
        for p in runs_root.rglob("*")
        if p.is_dir() and (p / "run-manifest.json").exists()
    ]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    for name in (
        "candidates.jsonl",
        "judgments.jsonl",
        "saves.jsonl",
        "run_log.jsonl",
        "progress.json",
        "run-report.json",
        "run-report.md",
        "runtime_state.sqlite3",
        SLATE_FILENAME,
    ):
        assert (run_dir / name).exists(), f"{name} missing from the snapshot"
    # ONE slate, not a glob of stale per-run siblings.
    assert [p.name for p in run_dir.glob("exa-slate*")] == [SLATE_FILENAME]

    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    assert manifest["source"] == "exa"
    assert int(manifest["run_id"]) == stats.run_id


def test_console_is_one_line_per_verdict(tmp_path: Path, capsys) -> None:
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "REVIEW_FLAGGED"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    asyncio.run(pipeline.run())

    out = capsys.readouterr().out
    verdicts = [
        line
        for line in out.splitlines()
        if line.startswith("    [exa] ")
        and line[10:].split(" ")[0] in {"SAVE", "REVIEW", "REJECT", "FAIL"}
    ]
    assert len(verdicts) == 2
    assert verdicts[0].startswith("    [exa] SAVE   Avery Holt — Investor, Benchmark")
    assert "(0.80)" in verdicts[0]
    assert "https://www.linkedin.com/in/averylinholt" in verdicts[0]
    assert verdicts[1].startswith("    [exa] REVIEW Mira Lorne — Principal, Greylock")
    # The console rule: no JSON, no multi-line blocks, no model prose.
    for line in out.splitlines():
        assert not line.lstrip().startswith("{")
        assert "\n" not in line


def test_person_key_is_written_on_candidate_rows(tmp_path: Path) -> None:
    """The store mints a person_key for github and nobody else.

    ``shared/runtime_state/store.py:1554`` is the whole special case, so
    an Exa row's key has to be passed in — a NULL here makes every
    cross-hub dedupe query blind to this person.
    """

    judge = ScriptedJudge({"Avery Holt": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    asyncio.run(pipeline.run())

    rows = _rows(
        tmp_path,
        "SELECT identity_key, person_key FROM candidates ORDER BY identity_key",
    )
    assert rows
    for row in rows:
        assert row["person_key"], f"{row['identity_key']} has a NULL person_key"
    # A public LinkedIn handle keys as linkedin:<handle> so an Exa find
    # and a LinkedIn find of the same human converge.
    assert {row["person_key"] for row in rows} == {
        "linkedin:averylinholt",
        "linkedin:mlorne",
    }


def test_known_from_sibling_reports_without_dropping_the_person(
    tmp_path: Path,
) -> None:
    """The cross-campaign check moves a slate line, never a decision.

    Avery Holt is still judged, still saved to the workspace, and still
    counted in ``stats.saved`` — she simply appears under "Already
    known" instead of the short list.
    """

    sibling = tmp_path / "sibling" / "9000000006"
    sibling.mkdir(parents=True)
    sibling_store = RuntimeStateStore(sibling / "runtime_state.sqlite3")
    sibling_run = sibling_store.start_run(
        source="linkedin",
        brief_id="linkedin-cos",
        output_dir=str(sibling),
        mode="fresh",
    )
    sibling_store.ensure_candidate(
        source="linkedin",
        brief_id="linkedin-cos",
        identity_key="avery-holt",
        display_name="Avery L. Holt",
        profile_url="https://www.linkedin.com/in/averylinholt",
    )
    for state in ("snippet_extracted", "full_started", "full_terminal"):
        sibling_store.set_candidate_state(
            run_id=sibling_run,
            source="linkedin",
            brief_id="linkedin-cos",
            identity_key="avery-holt",
            new_state=state,
            terminal_decision="SAVE" if state == "full_terminal" else None,
        )

    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "SAVE"})
    pipeline, _ = _pipeline(
        tmp_path, results=_TWO_FIRMS, judge=judge, known_from=[sibling]
    )

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Avery Holt", "Mira Lorne"], "known people are still judged"
    assert stats.saved == 2, "the save still landed"
    assert len(_rows(tmp_path, "SELECT * FROM workspace_entries")) == 2

    labels = {
        row["name"]: row["known_label"]
        for row in _jsonl(_state_dir(tmp_path) / "judgments.jsonl")
    }
    assert labels == {"Avery Holt": "9000000006", "Mira Lorne": ""}

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    known_section = slate.split("## Already known", 1)[1].split("## Coverage", 1)[0]
    assert "- **Avery Holt** — 9000000006" in known_section
    short_list = slate.split("## The short list", 1)[1].split("## Worth a look", 1)[0]
    assert "Mira Lorne" in short_list
    assert "Avery Holt" not in short_list
    # Both still carry the verify flag — an already-known title is as
    # stale as anybody else's.
    verify = slate.split("## Verify before outreach", 1)[1]
    assert "- **Avery Holt**" in verify
    assert "- **Mira Lorne**" in verify


def _recruiter_sibling(tmp_path: Path, names: list[str]) -> Path:
    """A sibling LinkedIn campaign whose saves carry Recruiter URLs only."""

    sibling = tmp_path / "sibling" / "9000000008"
    sibling.mkdir(parents=True)
    store = RuntimeStateStore(sibling / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="linkedin-ph", output_dir=str(sibling), mode="fresh"
    )
    for index, name in enumerate(names):
        key = f"recruiter-{index}"
        store.ensure_candidate(
            source="linkedin",
            brief_id="linkedin-ph",
            identity_key=key,
            display_name=name,
            profile_url=f"https://www.linkedin.com/talent/profile/AEMAA{index}",
        )
        for state in ("snippet_extracted", "full_started", "full_terminal"):
            store.set_candidate_state(
                run_id=run_id,
                source="linkedin",
                brief_id="linkedin-ph",
                identity_key=key,
                new_state=state,
                terminal_decision="SAVE" if state == "full_terminal" else None,
            )
    return sibling


def _already_known_section(tmp_path: Path) -> str:
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    return slate.split("## Already known", 1)[1].split("## Coverage", 1)[0]


def test_recruiter_saves_reach_the_already_known_section(tmp_path: Path) -> None:
    """CLO-330: a client slate said "Already known — 0" with 5 overlaps.

    The sibling's Recruiter rows have no public handle, so the Exa finds
    (which all have one) must fall back to the name, and "Mira L." must
    reach "Mira Lorne" through the initial.
    """

    sibling = _recruiter_sibling(tmp_path, ["Avery Holt", "Mira L."])
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge, known_from=[sibling])

    stats = asyncio.run(pipeline.run())

    assert stats.saved == 2, "known people are still judged and saved"
    known_section = _already_known_section(tmp_path)
    assert "- **Avery Holt** — 9000000008" in known_section
    assert "- **Mira Lorne** — 9000000008" in known_section


def test_an_initial_two_finds_fit_stays_off_the_already_known_section(
    tmp_path: Path,
) -> None:
    sibling = _recruiter_sibling(tmp_path, ["Mira L."])
    results = {
        "Benchmark": [_person("miralark", "Mira Lark", "Investor", "Benchmark")],
        "Greylock": [_person("mlorne", "Mira Lorne", "Principal", "Greylock")],
    }
    judge = ScriptedJudge({"Mira Lark": "SAVE", "Mira Lorne": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, results=results, judge=judge, known_from=[sibling])

    asyncio.run(pipeline.run())

    known_section = _already_known_section(tmp_path)
    assert "Mira Lark" not in known_section
    assert "Mira Lorne" not in known_section


def test_close_out_flags_degraded_run_when_half_the_judgments_fail(
    tmp_path: Path, capsys
) -> None:
    """A slate built on failed judgments must say so, out loud.

    A judgment failure is not a verdict about a person — the candidate
    stays retryable and the next session re-judges them — so a session
    where the transport was down still exits ``completed`` and still
    writes a slate. Without this line the operator reads that slate as
    a search result. The threshold is half; both people fail here, so
    the ratio is 1.0.
    """

    judge = ScriptedJudge({}, default="JUDGMENT_FAILURE")
    pipeline, _ = _pipeline(tmp_path, results=_TWO_FIRMS, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert stats.judged == 2
    assert stats.failed == 2
    assert _store(tmp_path).get_run(stats.run_id)["status"] == "completed"

    out = capsys.readouterr().out
    degraded = [line for line in out.splitlines() if "run degraded" in line]
    assert degraded == [
        "    [exa] run degraded — 2 of 2 judgments failed; "
        "verdicts are not trustworthy"
    ]
    # The slate the warning is about was still written — degradation is
    # a caveat on the artifacts, not a reason to withhold them.
    assert (_state_dir(tmp_path) / SLATE_FILENAME).exists()

    events = _rows(
        tmp_path,
        "SELECT payload_json FROM events WHERE event_type = 'run_degraded'",
    )
    assert len(events) == 1
    payload = json.loads(events[0]["payload_json"])
    assert payload["judged"] == 2
    assert payload["failed"] == 2
    assert payload["failure_ratio"] == 1.0
    assert payload["threshold"] == 0.5


def test_close_out_stays_quiet_below_the_degraded_threshold(
    tmp_path: Path, capsys
) -> None:
    """One failure in three is not a degraded run.

    The monitor exists to catch an outage, not to editorialize about a
    single bad call — a line that fires on ordinary flake trains the
    operator to ignore it. The threshold is ``>= 0.5``, so exactly half
    IS degraded; this fixture sits just under it deliberately.
    """

    judge = ScriptedJudge(
        {
            "Avery Holt": "SAVE",
            "Theo Marsh": "REJECT",
            "Quinn Avila": "JUDGMENT_FAILURE",
        }
    )
    pipeline, _ = _pipeline(tmp_path, results=_THREE_AT_BENCHMARK, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert (stats.judged, stats.failed) == (3, 1)
    assert "run degraded" not in capsys.readouterr().out
    assert not _rows(
        tmp_path,
        "SELECT payload_json FROM events WHERE event_type = 'run_degraded'",
    )


def test_close_out_does_not_flag_a_run_that_judged_nobody(
    tmp_path: Path, capsys
) -> None:
    """Zero judged is zero divided by zero, not a degraded run.

    A campaign whose queries returned nothing is a coverage problem the
    slate already reports; claiming its verdicts are untrustworthy when
    there are none would be a false alarm on every empty sweep.
    """

    pipeline, _ = _pipeline(tmp_path, results={"Benchmark": [], "Greylock": []})

    stats = asyncio.run(pipeline.run())

    assert stats.judged == 0
    assert "run degraded" not in capsys.readouterr().out


def test_brief_id_matches_universal_derivation(tmp_path: Path) -> None:
    """Store rows carry the brief identity every other plane uses.

    Two identities were being conflated. The state DIRECTORY is keyed by
    ``exa_state_key`` (brief.id → role_title → stem) — a per-source
    folder name. The brief IDENTITY written into ``runs``,
    ``candidates``, and the workspace idempotency key has to be
    ``derive_brief_id``'s, or Exa's rows for a brief are invisible to
    any query that resolves that brief the normal way. For the CoS brief
    that identity is the LinkedIn project id — which is exactly the join
    the "already known" bucket is built on.
    """

    from shared.brief_loader import load_brief
    from shared.output_paths import derive_brief_id, exa_state_key

    brief_path = _write_brief(
        tmp_path,
        role_title="Chief of Staff (CEO's Office)",
        source_config={"linkedin": {"project_id": "9000000006"}},
    )
    pipeline, _ = _pipeline(tmp_path, brief_path=brief_path)

    brief = load_brief(str(brief_path))
    assert pipeline.brief_id == derive_brief_id(
        brief_path=brief_path, brief=brief
    )
    assert pipeline.brief_id == "9000000006", (
        "the CoS brief's identity is its LinkedIn project id, not its role title"
    )

    # The state-dir key is a DIFFERENT thing and deliberately unchanged:
    # it stays the role-title slug, so the folder is still readable.
    assert exa_state_key(brief_path=brief_path, brief=brief) == (
        "chief_of_staff_ceo_s_office"
    )
    assert exa_state_key(brief_path=brief_path, brief=brief) != pipeline.brief_id


def test_brief_id_falls_back_when_no_project_id(tmp_path: Path) -> None:
    """A brief with no LinkedIn identity still gets a stable id.

    ``derive_brief_id`` falls through project_id → brief.id → stem, and
    ``brief.id`` on a V2 brief is the role title. Pinned so the change
    above is provably a widening, not a swap that breaks every brief
    that has no LinkedIn side.

    It also SLUGIFIES, which the old ``brief.id`` path did not — a
    brief with no LinkedIn side now identifies as ``standalone_exa_role``
    rather than ``Standalone Exa Role``. That is the point of using the
    universal derivation (every other plane already stores the slug),
    and it is stated here rather than discovered later by whoever
    wonders why one campaign's rows have a capital letter in them.
    """

    from shared.brief_loader import load_brief
    from shared.output_paths import derive_brief_id

    brief_path = _write_brief(tmp_path, role_title="Standalone Exa Role")
    pipeline, _ = _pipeline(tmp_path, brief_path=brief_path)

    assert pipeline.brief_id == derive_brief_id(
        brief_path=brief_path, brief=load_brief(str(brief_path))
    )
    assert pipeline.brief_id == "standalone_exa_role"


# ---------------------------------------------------------------------------
# S8c — a brief revised mid-campaign
# ---------------------------------------------------------------------------


_TWO_ARCHETYPES = [
    {
        "id": "a01-mbb-to-vc",
        "axis": "consulting_to_investing",
        "text": "Former McKinsey consultants in the SF Bay Area now investing.",
    },
    {
        "id": "a02-investor-to-cos",
        "axis": "investor_to_operator",
        "text": "Former SF Bay Area investors now Chief of Staff at an AI company.",
    },
]


def test_resume_runs_a_lane_the_brief_added(tmp_path: Path, capsys) -> None:
    """An archetype added between sessions actually searches.

    End to end, against the real store: session one runs the firm lane
    on a brief with no archetypes; session two resumes against a brief
    that has one, and the Exa client is asked for it. Before the plan
    merge the compiled plan was discarded on resume, so this query
    would never have been dispatched — the campaign ran the plan it was
    launched with until someone noticed.
    """

    one_lane = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        exa_query_lanes=["people_search"],
    )
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Dana Marlow": "SAVE"})
    first, _ = _pipeline(
        tmp_path, brief_path=one_lane, results=_TWO_FIRMS, judge=judge
    )
    asyncio.run(first.run())

    two_lanes = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        exa_query_lanes=["people_search", "archetype"],
        exa_archetype_queries=[_TWO_ARCHETYPES[0]],
    )
    results = {
        "Benchmark": _TWO_FIRMS["Benchmark"],
        "now investing": [
            _person("dmarlow", "Dana Marlow", "Principal", "Sutter Hill")
        ],
    }
    second, clients = _pipeline(
        tmp_path, brief_path=two_lanes, results=results, judge=judge
    )
    stats = asyncio.run(second.run(resume=True))

    assert clients[0].queries == [
        "Former McKinsey consultants in the SF Bay Area now investing."
    ], "the done firm query was not re-run; the new archetype was"
    assert judge.seen == ["Avery Holt", "Dana Marlow"]
    assert stats.candidates_discovered == 1

    rows = _store(tmp_path).list_work_units(stats.run_id, kind=EXA_QUERY_KIND)
    assert [(row["source_unit_id"], row["status"]) for row in rows] == [
        ("1", "done"),
        ("2", "done"),
    ]
    assert rows[1]["family_key"] == "archetype:a01-mbb-to-vc"
    assert "plan revised — 1 new query added" in capsys.readouterr().out


def test_resume_makes_no_exa_call_for_a_lane_the_brief_dropped(
    tmp_path: Path, capsys
) -> None:
    """A retired lane stops spending on the next session, not eventually.

    The stored ``queued`` rows used to outvote the edited brief: the
    operator deleted the lane, resumed, and watched it search anyway.
    Here session two must make exactly one Exa call — the firm query
    that was left over — and none for the two archetypes.
    """

    with_archetypes = _write_brief(
        tmp_path,
        exa_target_firms=list(_TWO_FIRM_LIST),
        exa_query_lanes=["people_search", "archetype"],
        exa_archetype_queries=list(_TWO_ARCHETYPES),
    )
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "SAVE"})
    first, _ = _pipeline(
        tmp_path, brief_path=with_archetypes, results=_TWO_FIRMS, judge=judge
    )
    asyncio.run(first.run(max_queries=1))

    firm_only = _write_brief(
        tmp_path,
        exa_target_firms=list(_TWO_FIRM_LIST),
        exa_query_lanes=["people_search"],
    )
    second, clients = _pipeline(
        tmp_path, brief_path=firm_only, results=_TWO_FIRMS, judge=judge
    )
    stats = asyncio.run(second.run(resume=True))

    assert clients[0].queries == [
        "Senior Associate or Principal at Greylock SF Bay Area"
    ], "an archetype whose lane the brief dropped was still searched"

    rows = {
        row["source_unit_id"]: row
        for row in _store(tmp_path).list_work_units(stats.run_id, kind=EXA_QUERY_KIND)
    }
    assert {key: row["status"] for key, row in rows.items()} == {
        "1": "done",
        "2": "done",
        "3": "skipped",
        "4": "skipped",
    }
    assert rows["3"]["notes"] == "lane no longer in brief"
    out = capsys.readouterr().out
    assert "plan revised — 2 queries skipped (lane dropped from the brief)" in out


def test_slate_notes_skipped_queries(tmp_path: Path) -> None:
    """A query nobody ran is reported as skipped, never as a gap.

    "Zero candidates" means the web did not surface anyone in band —
    the single most actionable line on the coverage list, because it
    tells a recruiter where the bench is empty. A retired query
    produces the same zero and means the opposite: nobody looked. The
    two must not render alike, and the retired one must not reach
    ``coverage_gaps`` in the structured report, where brief iteration
    would read it as a firm to widen.
    """

    both = _write_brief(
        tmp_path,
        exa_target_firms=list(_TWO_FIRM_LIST),
        exa_query_lanes=["people_search", "archetype"],
        exa_archetype_queries=list(_TWO_ARCHETYPES),
    )
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "SAVE"})
    first, _ = _pipeline(
        tmp_path, brief_path=both, results=_TWO_FIRMS, judge=judge
    )
    asyncio.run(first.run(max_queries=2))

    firm_only = _write_brief(
        tmp_path,
        exa_target_firms=list(_TWO_FIRM_LIST),
        exa_query_lanes=["people_search"],
    )
    second, _ = _pipeline(
        tmp_path, brief_path=firm_only, results=_TWO_FIRMS, judge=judge
    )
    stats = asyncio.run(second.run(resume=True))

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    coverage = slate.split("## Coverage notes", 1)[1].split(
        "## Verify before outreach", 1
    )[0]
    assert "Skipped — 2 queries (lane dropped from the brief)." in coverage
    # The archetype rows are still there, and they say they never ran
    # rather than that they found nobody.
    assert "Not run — the brief retired this query before it searched." in coverage
    assert "Rewrite the description before running it again." not in coverage

    report = json.loads((_state_dir(tmp_path) / "run-report.json").read_text())
    assert report["coverage_gaps"] == []
    performance = {row["firm"]: row for row in report["string_performance"]}
    assert performance["archetype:a01-mbb-to-vc"]["skipped"] == 1
    assert performance["Benchmark"]["skipped"] == 0
    # The two skipped queries never ran, so the campaign's query count
    # must not claim them.
    assert "2 queries returned" in slate.splitlines()[2]
    assert stats.queries_run == 0


def test_removing_an_archetype_does_not_shift_attribution(tmp_path: Path) -> None:
    """Placements follow the query, not the ordinal it happened to hold.

    Compiled ids are positional. Drop ``a01`` from the brief and the
    compiler renumbers ``a02`` from 2 to 1 — so a fresh run over the
    same state dir carries a work unit numbered 1 that is a02, while
    the campaign's candidates still say "found by query 1", meaning
    a01. Keyed on the id, a01's short list is reported as a02's, and
    the recruiter rewrites the wrong sentence. Keyed on
    ``(lane, coverage_key)``, which no renumbering touches, each
    hypothesis keeps its own people.
    """

    both = _write_brief(
        tmp_path,
        exa_target_firms=[],
        exa_query_lanes=["archetype"],
        exa_archetype_queries=list(_TWO_ARCHETYPES),
    )
    results = {
        "now investing": [
            _person("dmarlow", "Dana Marlow", "Principal", "Sutter Hill")
        ],
        "Chief of Staff at an AI company": [
            _person("ribarra", "Rafael Ibarra", "Chief of Staff", "Rowan AI")
        ],
    }
    judge = ScriptedJudge(
        {"Dana Marlow": "SAVE", "Rafael Ibarra": "REVIEW_FLAGGED"}
    )
    first, _ = _pipeline(tmp_path, brief_path=both, results=results, judge=judge)
    asyncio.run(first.run())

    rows = _rows(
        tmp_path,
        "SELECT identity_key, source_unit_id FROM candidates c "
        "JOIN work_units w ON w.id = c.last_work_unit_id ORDER BY c.id",
    )
    assert [row["source_unit_id"] for row in rows] == ["1", "2"]

    # a01 is retired. A FRESH run compiles a02 as query 1 — the id Dana
    # is stored under.
    a02_only = _write_brief(
        tmp_path,
        exa_target_firms=[],
        exa_query_lanes=["archetype"],
        exa_archetype_queries=[_TWO_ARCHETYPES[1]],
    )
    second, _ = _pipeline(
        tmp_path, brief_path=a02_only, results=results, judge=judge
    )
    stats = asyncio.run(second.run())

    assert judge.seen == ["Dana Marlow", "Rafael Ibarra"], "nobody was re-judged"

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    archetypes = slate.split("### Archetype queries", 1)[1]
    a02_row = next(
        line for line in archetypes.splitlines() if "a02-investor-to-cos" in line
    )
    assert "0 short list, 1 worth a look" in a02_row, (
        "a01's short list was credited to a02 by ordinal"
    )
    assert "a01-mbb-to-vc" not in archetypes, "the retired query is not in this run"
    assert stats.run_id


# ---------------------------------------------------------------------------
# Resume re-runs a query a transient network fault errored (2026-10-10)
# ---------------------------------------------------------------------------


def _dns_error():
    import socket
    import types

    import aiohttp

    key = types.SimpleNamespace(host="api.exa.ai", port=443, ssl=True, is_ssl=True)
    return aiohttp.ClientConnectorDNSError(
        key, socket.gaierror(8, "nodename nor servname provided, or not known")
    )


class _FirstCallFaultClient(FakeExaClient):
    """Raises ``error`` on its first search only, before any dispatch."""

    def __init__(self, meter, *, results, error):
        super().__init__(meter, results=results)
        self._fault = error

    async def search(self, query: str, **kwargs):
        if self._fault is not None:
            fault, self._fault = self._fault, None
            raise fault
        return await super().search(query, **kwargs)


def _first_call_fault_pipeline(tmp_path, *, brief_path, judge, error):
    pipeline = ExaPipeline(
        brief_path=brief_path,
        output_dir=_state_dir(tmp_path),
        client_factory=lambda meter: _FirstCallFaultClient(
            meter, results=_TWO_FIRMS, error=error
        ),
        judge=judge,
        extractor=_fake_extractor,
    )
    return pipeline


def test_resume_reruns_a_query_a_dns_outage_errored(tmp_path: Path, capsys) -> None:
    """The SPL run's six DNS-errored archetypes must run again on resume."""

    brief_path = _write_brief(tmp_path)
    judge = ScriptedJudge({"Avery Holt": "SAVE", "Mira Lorne": "SAVE"})

    first = _first_call_fault_pipeline(
        tmp_path, brief_path=brief_path, judge=judge, error=_dns_error()
    )
    stats_one = asyncio.run(first.run())
    store = _store(tmp_path)
    rows = store.list_work_units(stats_one.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in rows] == ["error", "done"]
    assert "ClientConnectorDNSError" in rows[0]["notes"]
    capsys.readouterr()

    second, clients_two = _pipeline(
        tmp_path, brief_path=brief_path, results=_TWO_FIRMS, judge=judge
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert clients_two[0].queries == [
        "Senior Associate or Principal at Benchmark SF Bay Area"
    ], "the errored query ran again and the done one did not"
    assert judge.seen == ["Mira Lorne", "Avery Holt"]
    rows = store.list_work_units(stats_two.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in rows] == ["done", "done"]
    assert rows[0]["notes"] == ""
    out = capsys.readouterr().out
    assert "1 query requeued after a transient network error" in out


def test_resume_leaves_a_query_a_bad_request_errored(tmp_path: Path) -> None:
    """A 4xx is the query's own fault; resuming would only repeat it."""

    brief_path = _write_brief(tmp_path)
    judge = ScriptedJudge({})
    first = _first_call_fault_pipeline(
        tmp_path,
        brief_path=brief_path,
        judge=judge,
        error=RuntimeError("Exa API error 400 (INVALID_REQUEST): bad query"),
    )
    asyncio.run(first.run())

    second, clients_two = _pipeline(
        tmp_path, brief_path=brief_path, results=_TWO_FIRMS, judge=judge
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert clients_two == [] or clients_two[0].queries == []
    rows = _store(tmp_path).list_work_units(stats_two.run_id, kind=EXA_QUERY_KIND)
    assert [row["status"] for row in rows] == ["error", "done"]


@pytest.mark.parametrize(
    "note",
    [
        "exa search failed: ClientConnectorDNSError: Cannot connect to host api.exa.ai:443 ssl:default [nodename nor servname provided, or not known]",
        "exa search failed: ServerDisconnectedError: Server disconnected",
        "exa search failed: TimeoutError: ",
        "exa search failed: ConnectionTimeoutError: Connection timeout to host https://api.exa.ai/search",
        "exa search failed: RuntimeError: Exa API request failed after 5 retries: https://api.exa.ai/search",
    ],
)
def test_transient_query_notes_are_recognized(note: str) -> None:
    from exa.acquisition import is_transient_query_failure

    assert is_transient_query_failure(note) is True


@pytest.mark.parametrize(
    "note",
    [
        "",
        "exa search failed: RuntimeError: Exa API error 400 (INVALID_REQUEST): bad",
        "exa search failed: RuntimeError: Exa API error 404: not found",
        "lane no longer in brief",
        "something else entirely",
    ],
)
def test_non_transient_query_notes_are_not(note: str) -> None:
    from exa.acquisition import is_transient_query_failure

    assert is_transient_query_failure(note) is False


# ---------------------------------------------------------------------------
# Resume re-judges slate verdicts the brief's criteria have moved past
# (2026-10-10 review: after a brief revision, the audit's bad saves stayed
# on the short list because a stored verdict was never judged again)
# ---------------------------------------------------------------------------


def _section(slate: str, heading: str) -> str:
    """The body of one ``## heading`` section of the slate."""

    if heading not in slate:
        return ""
    return slate.split(heading, 1)[1].split("\n## ", 1)[0]


def _terminal_states(tmp_path: Path) -> dict[str, tuple[str, str | None]]:
    return {
        row["identity_key"]: (row["current_lifecycle_state"], row["terminal_decision"])
        for row in _rows(
            tmp_path,
            "SELECT identity_key, current_lifecycle_state, terminal_decision "
            "FROM candidates",
        )
    }


def _first_session(tmp_path: Path, verdicts: dict[str, str]) -> Path:
    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    first, _ = _pipeline(
        tmp_path,
        brief_path=brief,
        results=_THREE_AT_BENCHMARK,
        judge=ScriptedJudge(verdicts),
    )
    asyncio.run(first.run())
    return brief


def test_resume_rejudges_slate_verdicts_after_the_judging_criteria_change(
    tmp_path: Path, capsys
) -> None:
    """A revised bar reaches the people already on the slate.

    Session one saves Avery and puts Theo on the worth-a-look list. The
    brief's instructions then change, which changes what the judge is
    told. On resume both are judged again, with no Exa call: Avery now
    rejects and leaves the short list, Theo now saves and joins it.
    Quinn was rejected under the old criteria and is not on the slate,
    so she is not re-judged.
    """

    _first_session(
        tmp_path,
        {"Avery Holt": "SAVE", "Theo Marsh": "REVIEW_FLAGGED"},
    )
    assert len(_rows(tmp_path, "SELECT * FROM workspace_entries")) == 1
    capsys.readouterr()

    revised = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        instructions=["The person must be able to read code."],
    )
    judge_two = ScriptedJudge(
        {"Avery Holt": "REJECT", "Theo Marsh": "SAVE"}, repeats=2
    )
    second, clients_two = _pipeline(
        tmp_path, brief_path=revised, results=_THREE_AT_BENCHMARK, judge=judge_two,
        rejudge_stale=True,
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert clients_two == [], "re-judging costs no Exa call"
    assert sorted(judge_two.seen) == [
        "Avery Holt", "Avery Holt", "Theo Marsh", "Theo Marsh"
    ], "two samples per slate verdict"
    states = _terminal_states(tmp_path)
    assert states["linkedin.com/in/averylinholt"] == ("full_terminal", "REJECT")
    assert states["linkedin.com/in/theomarsh"] == ("full_terminal", "SAVE")
    assert states["linkedin.com/in/quinnxavila"] == ("full_terminal", "REJECT")

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    short_list = _section(slate, "## The short list")
    assert "Theo Marsh" in short_list
    assert "Avery Holt" not in short_list
    assert "Avery Holt" not in _section(slate, "## Worth a look")

    assert stats_two.rejudged == 2
    assert stats_two.rejudge_left_short_list == 1

    # The verdict the store no longer holds survives as an event.
    events = [
        json.loads(row["payload_json"])
        for row in _rows(
            tmp_path,
            "SELECT payload_json FROM events WHERE event_type = ?",
            ("exa_stale_verdict_rejudged",),
        )
    ]
    old = {event["identity_key"]: event["old_decision"]["decision"] for event in events}
    assert old == {
        "linkedin.com/in/averylinholt": "SAVE",
        "linkedin.com/in/theomarsh": "REVIEW_FLAGGED",
    }

    # Theo's new save lands in the workspace once; Avery's row stays.
    assert len(_rows(tmp_path, "SELECT * FROM workspace_entries")) == 2
    out = capsys.readouterr().out
    assert (
        "[exa] re-judging 2 slate verdicts (two samples each) made under earlier "
        "judging criteria, 16 at a time (no Exa call)"
    ) in out

    # A third session under the same brief re-judges nobody.
    judge_three = ScriptedJudge({})
    third, _ = _pipeline(
        tmp_path, brief_path=revised, results=_THREE_AT_BENCHMARK, judge=judge_three,
        rejudge_stale=True,
    )
    stats_three = asyncio.run(third.run(resume=True))
    assert judge_three.seen == []
    assert stats_three.rejudged == 0


def test_resume_does_not_rejudge_when_only_the_search_plan_changed(
    tmp_path: Path,
) -> None:
    """A brief edit the judge never sees is not a reason to judge again."""

    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    plan_only = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        _exa_brief_notes="search-plan note the judge never reads",
    )
    judge_two = ScriptedJudge({"Avery Holt": "REJECT"})
    second, _ = _pipeline(
        tmp_path, brief_path=plan_only, results=_THREE_AT_BENCHMARK, judge=judge_two,
        rejudge_stale=True,
    )
    stats_two = asyncio.run(second.run(resume=True))

    assert judge_two.seen == []
    assert stats_two.rejudged == 0
    assert _terminal_states(tmp_path)["linkedin.com/in/averylinholt"] == (
        "full_terminal",
        "SAVE",
    )


def test_resume_rejudges_a_verdict_with_no_fingerprint_when_the_brief_changed(
    tmp_path: Path,
) -> None:
    """Verdicts written before the fingerprint existed fall back to the brief hash.

    The live SPL campaign's 190 saves carry no criteria fingerprint. Any
    change to the brief file is then the only evidence the criteria may
    have moved, so the verdict is judged again, once.
    """

    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    conn = sqlite3.connect(_state_dir(tmp_path) / "runtime_state.sqlite3")
    try:
        conn.execute(
            "UPDATE candidates SET terminal_payload_json = "
            "json_remove(terminal_payload_json, '$.judgment_fingerprint')"
        )
        conn.commit()
    finally:
        conn.close()

    unchanged = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST)),
        results=_THREE_AT_BENCHMARK,
        judge=ScriptedJudge({}),
        rejudge_stale=True,
    )[0]
    assert asyncio.run(unchanged.run(resume=True)).rejudged == 0, (
        "an unchanged brief file never re-judges"
    )

    edited = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        _exa_brief_notes="any edit at all",
    )
    judge = ScriptedJudge({"Avery Holt": "REJECT"}, repeats=2)
    second, _ = _pipeline(
        tmp_path, brief_path=edited, results=_THREE_AT_BENCHMARK, judge=judge,
        rejudge_stale=True,
    )
    stats = asyncio.run(second.run(resume=True))

    assert judge.seen == ["Avery Holt", "Avery Holt"]
    assert stats.rejudged == 1
    assert _terminal_states(tmp_path)["linkedin.com/in/averylinholt"] == (
        "full_terminal",
        "REJECT",
    )


def test_resume_never_rejudges_a_verdict_a_recruiter_decided(tmp_path: Path) -> None:
    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    conn = sqlite3.connect(_state_dir(tmp_path) / "runtime_state.sqlite3")
    try:
        conn.execute("UPDATE candidates SET user_status = 'confirmed'")
        conn.commit()
    finally:
        conn.close()

    revised = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        instructions=["The person must be able to read code."],
    )
    judge = ScriptedJudge({"Avery Holt": "REJECT"})
    second, _ = _pipeline(
        tmp_path, brief_path=revised, results=_THREE_AT_BENCHMARK, judge=judge,
        rejudge_stale=True,
    )
    asyncio.run(second.run(resume=True))

    assert judge.seen == []
    assert _terminal_states(tmp_path)["linkedin.com/in/averylinholt"] == (
        "full_terminal",
        "SAVE",
    )


def test_a_rejudge_interrupted_after_the_clear_finishes_next_session(
    tmp_path: Path, monkeypatch
) -> None:
    """The clear and the new verdict are two writes; a death between them heals.

    A cleared verdict is no longer terminal and not yet pending. Without
    an explicit pick-up it would vanish from the slate and never be
    judged again.
    """

    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    revised = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        instructions=["The person must be able to read code."],
    )

    class _Killed(RuntimeError):
        pass

    original = RuntimeStateStore.clear_candidate_terminal_state

    def _clear_then_die(self, **kwargs):
        original(self, **kwargs)
        raise _Killed("process killed after the clear")

    monkeypatch.setattr(
        RuntimeStateStore, "clear_candidate_terminal_state", _clear_then_die
    )
    second, _ = _pipeline(
        tmp_path, brief_path=revised, results=_THREE_AT_BENCHMARK,
        judge=ScriptedJudge({}, repeats=2), rejudge_stale=True,
    )
    with pytest.raises(_Killed):
        asyncio.run(second.run(resume=True))
    monkeypatch.setattr(RuntimeStateStore, "clear_candidate_terminal_state", original)
    assert _terminal_states(tmp_path)["linkedin.com/in/averylinholt"] == (
        "full_terminal",
        None,
    )

    # Without the opt-in: a cleared row holds no verdict, so it is always
    # finished, whatever the campaign chose about stale verdicts.
    judge = ScriptedJudge({"Avery Holt": "INFERENTIAL_SAVE"}, repeats=2)
    third, _ = _pipeline(
        tmp_path, brief_path=revised, results=_THREE_AT_BENCHMARK, judge=judge
    )
    stats = asyncio.run(third.run(resume=True))

    # The cleared row's old verdict (SAVE) is read back from the event, so it
    # is judged as a slate verdict: two samples.
    assert judge.seen == ["Avery Holt", "Avery Holt"]
    assert stats.rejudged == 1
    assert _terminal_states(tmp_path)["linkedin.com/in/averylinholt"] == (
        "full_terminal",
        "INFERENTIAL_SAVE",
    )
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Avery Holt" in _section(slate, "## The short list")


def test_a_stranded_save_the_new_criteria_reject_never_lands(tmp_path: Path) -> None:
    """The re-judge runs before the interrupted-save replay.

    Session one dies between Avery's SAVE and its workspace write. The
    criteria then change and now reject her. Replaying first would land
    her in the workspace under a verdict the current brief no longer
    makes; re-judging first means she never lands.
    """

    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    first, _ = _pipeline(
        tmp_path,
        brief_path=brief,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=ScriptedJudge({"Avery Holt": "SAVE"}),
    )

    from shared.save_destination.candidate_workspace import (
        CandidateWorkspaceSaveDestination,
    )

    class _Killed(RuntimeError):
        pass

    original = CandidateWorkspaceSaveDestination.save
    try:
        def _die(self, **kwargs):
            raise _Killed("process killed mid-save")

        CandidateWorkspaceSaveDestination.save = _die
        with pytest.raises(_Killed):
            asyncio.run(first.run())
    finally:
        CandidateWorkspaceSaveDestination.save = original
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []

    revised = _write_brief(
        tmp_path,
        exa_target_firms=list(_ONE_FIRM_LIST),
        instructions=["The person must be able to read code."],
    )
    judge = ScriptedJudge({"Avery Holt": "REJECT"}, repeats=2)
    second, _ = _pipeline(
        tmp_path,
        brief_path=revised,
        results={"Benchmark": _THREE_AT_BENCHMARK["Benchmark"][:1]},
        judge=judge,
        rejudge_stale=True,
    )
    stats = asyncio.run(second.run(resume=True))

    assert judge.seen == ["Avery Holt", "Avery Holt"]
    assert stats.replayed_saves == 0
    assert stats.rejudge_left_short_list == 1
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []
    assert not (_state_dir(tmp_path) / "saves.jsonl").exists()


# ---------------------------------------------------------------------------
# The re-judge never ratchets silently or on one noisy sample (2026-10-10
# review round 3): two concurring rejects to leave the slate, a split goes to
# worth a look, every departure is listed, saves.jsonl agrees with the slate,
# and earlier rejects are re-judged only on the campaign's opt-in.
# ---------------------------------------------------------------------------


def _revised(tmp_path: Path, **overrides) -> Path:
    fields = {
        "exa_target_firms": list(_ONE_FIRM_LIST),
        "instructions": ["The person must be able to read code."],
        **overrides,
    }
    return _write_brief(tmp_path, **fields)


def _rejudge(tmp_path: Path, judge, *, brief: Path | None = None, **pipeline_kwargs):
    pipeline, clients = _pipeline(
        tmp_path,
        brief_path=brief or _revised(tmp_path),
        results=_THREE_AT_BENCHMARK,
        judge=judge,
        **{"rejudge_stale": True, **pipeline_kwargs},
    )
    stats = asyncio.run(pipeline.run(resume=True))
    return stats, clients


def _saves(tmp_path: Path, identity_key: str) -> list[dict]:
    return [
        row
        for row in _jsonl(_state_dir(tmp_path) / "saves.jsonl")
        if row["identity_key"] == identity_key
    ]


AVERY = "linkedin.com/in/averylinholt"
SEAN = "linkedin.com/in/theomarsh"
QUINN = "linkedin.com/in/quinnxavila"


def test_one_reject_sample_never_takes_a_save_off_the_slate(tmp_path: Path) -> None:
    """A split (one save-type sample, one reject) lands on worth-a-look.

    The stored decision is the save-type sample, the payload carries both,
    the slate files the person under worth-a-look with the reject sample's
    reason, and the left-the-short-list section names the move.
    """

    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    judge = ScriptedJudge(
        {}, repeats=2, sequences={"Avery Holt": ["REJECT", "INFERENTIAL_SAVE"]}
    )
    stats, clients = _rejudge(tmp_path, judge)

    assert clients == []
    assert judge.seen == ["Avery Holt", "Avery Holt"]
    assert stats.rejudge_split == 1
    assert stats.rejudge_left_short_list == 1
    assert stats.review == 1 and stats.rejected == 0
    assert _terminal_states(tmp_path)[AVERY] == ("full_terminal", "INFERENTIAL_SAVE")
    payload = json.loads(
        _rows(tmp_path, "SELECT terminal_payload_json FROM candidates WHERE identity_key = ?", (AVERY,))[0][0]
    )
    assert sorted(s["decision"] for s in payload["rejudge_split"]["samples"]) == [
        "INFERENTIAL_SAVE", "REJECT"
    ]

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Avery Holt" not in _section(slate, "## The short list")
    look = _section(slate, "## Worth a look")
    assert "Avery Holt" in look
    assert "re-judge split: one sample INFERENTIAL_SAVE, one REJECT" in look
    left = _section(slate, "## Left the short list on re-judge — 1")
    assert "**Avery Holt**" in left
    assert (
        "Was SAVE (0.80); now worth a look: re-judge split: one sample "
        "INFERENTIAL_SAVE, one REJECT: Avery Holt: scripted rationale."
    ) in left
    marker = _saves(tmp_path, AVERY)[-1]
    assert marker["status"] == "superseded"
    assert marker["rejudge"]["now"] == "worth a look"
    assert marker["rejudge"]["reason"].startswith("re-judge split: one sample INFERENTIAL_SAVE")

    rows = {row["name"]: row for row in _jsonl(_state_dir(tmp_path) / "judgments.jsonl") if row["run_id"] == 2}
    assert rows["Avery Holt"]["bucket"] == "review"
    assert rows["Avery Holt"]["rejudge"] == "split"

    # The split is worth a look, not an interrupted save: the next session
    # never replays it into the workspace.
    later, _ = _pipeline(
        tmp_path, brief_path=_revised(tmp_path), results=_THREE_AT_BENCHMARK,
        judge=ScriptedJudge({}),
    )
    assert asyncio.run(later.run(resume=True)).replayed_saves == 0


def test_two_concurring_rejects_take_a_save_off_and_saves_jsonl_agrees(
    tmp_path: Path,
) -> None:
    _first_session(tmp_path, {"Avery Holt": "SAVE", "Theo Marsh": "SAVE"})
    judge = ScriptedJudge({"Avery Holt": "REJECT", "Theo Marsh": "SAVE"}, repeats=2)
    stats, _ = _rejudge(tmp_path, judge)

    assert stats.rejudge_left_short_list == 1
    assert stats.rejudge_split == 0
    assert _terminal_states(tmp_path)[AVERY] == ("full_terminal", "REJECT")

    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    left = _section(slate, "## Left the short list on re-judge — 1")
    assert (
        "- **Avery Holt** — Investor, Benchmark (San Francisco, California). Was SAVE "
        "(0.80); now rejected: Avery Holt: scripted rationale."
    ) in left
    assert "Theo Marsh" not in left

    # The original save line stays; a later line marks it superseded.
    avery = _saves(tmp_path, AVERY)
    assert [row["status"] for row in avery] == ["succeeded", "superseded"]
    assert avery[-1]["decision"] == "REJECT"
    assert avery[-1]["rejudge"]["previous_decision"] == "SAVE"
    assert avery[-1]["rejudge"]["now"] == "rejected"
    assert [row["status"] for row in _saves(tmp_path, SEAN)] == ["succeeded"]
    # Avery keeps her workspace row; the recruiter's fields live there.
    assert len(_rows(tmp_path, "SELECT * FROM workspace_entries")) == 2


def test_a_failed_sample_leaves_the_stored_verdict_untouched(tmp_path: Path) -> None:
    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    judge = ScriptedJudge(
        {}, repeats=2, sequences={"Avery Holt": ["PARSE_FAILURE", "REJECT"]}
    )
    stats, _ = _rejudge(tmp_path, judge)

    assert stats.rejudge_kept == 1
    assert stats.rejudged == 0
    assert _terminal_states(tmp_path)[AVERY] == ("full_terminal", "SAVE")
    assert _rows(
        tmp_path, "SELECT * FROM events WHERE event_type = 'candidate_terminal_cleared'"
    ) == []
    kept = _rows(
        tmp_path,
        "SELECT payload_json FROM events WHERE event_type = 'exa_stale_verdict_rejudge_kept'",
    )
    assert json.loads(kept[0][0])["reason"] == "sample_failed"
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Avery Holt" in _section(slate, "## The short list")
    assert [row["status"] for row in _saves(tmp_path, AVERY)] == ["succeeded"]

    # Still stale, so the next resume tries again.
    retry = ScriptedJudge({"Avery Holt": "SAVE"}, repeats=2)
    again, _ = _rejudge(tmp_path, retry)
    assert retry.seen == ["Avery Holt", "Avery Holt"]
    assert again.rejudged == 1


def test_stale_rejects_are_rejudged_only_on_the_opt_in(tmp_path: Path) -> None:
    """One save-type sample promotes an earlier reject; without the opt-in nobody is."""

    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    off = ScriptedJudge({"Quinn Avila": "SAVE"}, repeats=2)
    stats_off, _ = _rejudge(tmp_path, off)
    assert "Quinn Avila" not in off.seen
    assert stats_off.rejudge_rejects == 0

    on = ScriptedJudge({"Quinn Avila": "SAVE", "Theo Marsh": "REJECT"})
    stats_on, clients = _rejudge(
        tmp_path,
        on,
        brief=_revised(tmp_path, _exa_brief_notes="a second edit"),
        rejudge_stale_rejects=True,
    )

    assert clients == []
    assert sorted(on.seen) == ["Quinn Avila", "Theo Marsh"], "one sample per reject"
    assert stats_on.rejudge_rejects == 2
    assert stats_on.rejudge_promoted == 1
    assert _terminal_states(tmp_path)[QUINN] == ("full_terminal", "SAVE")
    assert _terminal_states(tmp_path)[SEAN] == ("full_terminal", "REJECT")
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Quinn Avila" in _section(slate, "## The short list")
    assert [row["status"] for row in _saves(tmp_path, QUINN)] == ["succeeded"]


def test_rejudge_samples_run_in_parallel(tmp_path: Path) -> None:
    """Four samples meet at one barrier: a serial pass would time out on it."""

    _first_session(tmp_path, {"Avery Holt": "SAVE", "Theo Marsh": "SAVE"})
    barrier = threading.Barrier(4, timeout=10)

    class _Gathered(ScriptedJudge):
        def __call__(self, candidate, brief, *, query=None):
            barrier.wait()
            return super().__call__(candidate, brief, query=query)

    stats, _ = _rejudge(tmp_path, _Gathered({}, default="SAVE", repeats=2))
    assert stats.rejudged == 2


def test_a_later_rejudge_that_restores_a_save_reinstates_it(tmp_path: Path) -> None:
    _first_session(tmp_path, {"Avery Holt": "SAVE"})
    _rejudge(tmp_path, ScriptedJudge({"Avery Holt": "REJECT"}, repeats=2))
    assert [row["status"] for row in _saves(tmp_path, AVERY)] == ["succeeded", "superseded"]

    restored = ScriptedJudge({"Avery Holt": "SAVE"}, repeats=2)
    _rejudge(
        tmp_path,
        restored,
        brief=_revised(tmp_path, instructions=["Reading code is the bar; say which basis."]),
        rejudge_stale_rejects=True,
    )

    avery = _saves(tmp_path, AVERY)
    assert [row["status"] for row in avery] == ["succeeded", "superseded", "succeeded"]
    assert avery[-1]["reinstated"] is True
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Avery Holt" in _section(slate, "## The short list")
    assert "## Left the short list on re-judge" not in slate


def test_a_location_exclusion_on_rejudge_is_counted_apart(tmp_path: Path) -> None:
    """The location gate settles a stale save with no model call, and is not
    counted as the new criteria moving someone off the short list."""

    abroad = {
        "Benchmark": [
            _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
            FakeResult(
                url="https://www.linkedin.com/in/nikovarga",
                title="Niko Varga - Investor - Benchmark | LinkedIn",
                text="Hyderabad, Telangana, India\nNiko Varga works at Benchmark.",
                highlights=["Niko Varga is an Investor at Benchmark."],
            ),
        ]
    }
    brief = _write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST))
    first, _ = _pipeline(
        tmp_path, brief_path=brief, results=abroad,
        judge=ScriptedJudge({"Avery Holt": "SAVE", "Niko Varga": "SAVE"}),
    )
    asyncio.run(first.run())

    judge = ScriptedJudge({"Avery Holt": "REJECT"}, repeats=2)
    second, _ = _pipeline(
        tmp_path,
        brief_path=_revised(tmp_path, exa_location_scope={"countries": ["United States"]}),
        results=abroad,
        judge=judge,
        rejudge_stale=True,
    )
    stats = asyncio.run(second.run(resume=True))

    assert judge.seen == ["Avery Holt", "Avery Holt"], "the gate needs no model call"
    assert stats.rejudged == 2
    assert stats.rejudge_left_short_list == 1
    assert stats.rejudge_location_excluded == 1
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    left = _section(slate, "## Left the short list on re-judge — 2")
    assert "1 on the judge's verdict" in left and "1 excluded by a gate" in left
    judged, gated = left.split("Excluded by a gate on re-judge — 1:")
    assert "**Avery Holt**" in judged and "**Niko Varga**" in gated
    assert (
        "Was SAVE (0.80); now excluded (location scope): located outside the United "
        "States (Hyderabad, Telangana, India). https://www.linkedin.com/in/nikovarga"
    ) in left
    assert [row["status"] for row in _saves(tmp_path, "linkedin.com/in/nikovarga")] == [
        "succeeded", "superseded"
    ]


def test_a_split_that_never_reached_the_workspace_is_not_replayed(tmp_path: Path) -> None:
    """A worth-a-look verdict that splits keeps the save sample's decision.

    Theo was never written to the workspace (he was worth a look). After the
    split his stored decision is SAVE, which the interrupted-save replay
    would read as a save that never landed; it must not land him.
    """

    _first_session(tmp_path, {"Theo Marsh": "REVIEW_FLAGGED"})
    judge = ScriptedJudge({}, repeats=2, sequences={"Theo Marsh": ["SAVE", "REJECT"]})
    stats, _ = _rejudge(tmp_path, judge)
    assert stats.rejudge_split == 1
    assert stats.rejudge_left_short_list == 0, "he was never on the short list"
    assert _terminal_states(tmp_path)[SEAN] == ("full_terminal", "SAVE")
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []

    later, _ = _pipeline(
        tmp_path, brief_path=_revised(tmp_path), results=_THREE_AT_BENCHMARK,
        judge=ScriptedJudge({}),
    )
    assert asyncio.run(later.run(resume=True)).replayed_saves == 0
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Theo Marsh" in _section(slate, "## Worth a look")
    assert "## Left the short list on re-judge" not in slate


# ---------------------------------------------------------------------------
# The re-judge pass is the campaign's opt-in (review round 3): any lane's
# next resume used to re-judge every legacy verdict whose brief file changed
# (131 and 26 verdicts on two live campaigns).
# ---------------------------------------------------------------------------


def test_a_resume_without_the_opt_in_rejudges_nothing(tmp_path: Path, capsys) -> None:
    _first_session(
        tmp_path,
        {"Avery Holt": "SAVE", "Theo Marsh": "REVIEW_FLAGGED"},
    )
    before = _terminal_states(tmp_path)
    capsys.readouterr()

    judge = ScriptedJudge({}, default="REJECT")
    pipeline, clients = _pipeline(
        tmp_path, brief_path=_revised(tmp_path), results=_THREE_AT_BENCHMARK, judge=judge
    )
    stats = asyncio.run(pipeline.run(resume=True))

    assert judge.seen == []
    assert clients == []
    assert stats.rejudged == 0
    assert _terminal_states(tmp_path) == before
    assert _rows(
        tmp_path,
        "SELECT * FROM events WHERE event_type IN "
        "('exa_stale_verdict_rejudged', 'candidate_terminal_cleared')",
    ) == []
    assert "re-judging" not in capsys.readouterr().out



def test_departures_are_measured_against_the_slate_the_recruiter_saw(
    tmp_path: Path,
) -> None:
    """A save the scope check had already moved to worth-a-look never left the
    short list, and an earlier reject a gate settles was never on the slate."""

    from exa.scope_gate import SCOPE_FIELD

    people = {
        "Benchmark": [
            _person("annacme", "Ann Acme", "Investor", "Acme Corp"),
            FakeResult(
                url="https://www.linkedin.com/in/nikovarga",
                title="Niko Varga - Investor - Benchmark | LinkedIn",
                text="Hyderabad, Telangana, India\nNiko Varga works at Benchmark.",
                highlights=["Niko Varga is an Investor at Benchmark."],
            ),
        ]
    }
    scope = {SCOPE_FIELD: {"aliases": {"Outlier": "Benchmark"}}}
    first, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, exa_target_firms=list(_ONE_FIRM_LIST), **scope),
        results=people,
        judge=ScriptedJudge({"Ann Acme": "SAVE", "Niko Varga": "REJECT"}),
    )
    asyncio.run(first.run())
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "Ann Acme" in _section(slate, "## Worth a look"), "the scope moved her"

    second, _ = _pipeline(
        tmp_path,
        brief_path=_revised(
            tmp_path, exa_location_scope={"countries": ["United States"]}, **scope
        ),
        results=people,
        judge=ScriptedJudge({"Ann Acme": "REJECT"}, repeats=2),
        rejudge_stale=True,
        rejudge_stale_rejects=True,
    )
    stats = asyncio.run(second.run(resume=True))

    assert stats.rejudged == 2
    assert stats.rejudge_left_short_list == 0
    assert stats.rejudge_location_excluded == 0, "Niko was a reject, never on the slate"
    slate = (_state_dir(tmp_path) / SLATE_FILENAME).read_text()
    assert "## Left the short list on re-judge" not in slate
