"""Exa current-employer scope check: a post-judgment, non-destructive rule.

A competitors-only brief restricts saves to people whose CURRENT employer is
one of its target firms. The judge carries that rule as prompt text, and a
judge slip or a stale page can still put someone at another company on the
short list. The scope check reads the record's current employer after the
verdict:

* a save whose current employer is positively identified and is not in scope
  moves to the review bucket, saying "scope: current employer X";
* a record with no identifiable employer keeps its verdict;
* nobody is dropped before judgment, and the stored verdict is never changed;
* a brief that does not carry ``exa_current_employer_scope`` is untouched.

Scope is every target firm's name and alias, plus the block's own aliases
(contractor brands mapped to a parent), which never become query terms.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from exa.lane_compiler import blocking_findings, compile_exa_queries, lint_exa_brief
from exa.schemas import ExaCandidate, ExaEvidence
from exa.scope_gate import (
    IN_SCOPE,
    OUT_OF_SCOPE,
    SCOPE_FIELD,
    UNIDENTIFIED,
    CurrentEmployerScope,
    scope_errors,
)
from shared.brief_loader import load_brief
from tests.test_exa_orchestrator import (
    FakeResult,
    ScriptedJudge,
    _jsonl,
    _person,
    _pipeline,
    _rows,
    _state_dir,
    _write_brief,
)

FIRMS = [
    {"name": "Benchmark", "domain": "benchmark.com", "aliases": ["Benchmark Capital"]},
    {"name": "Greylock", "domain": "greylock.com"},
]
SCOPE = {"aliases": {"Outlier": "Benchmark", "Remotasks": "Benchmark"}}


def _scoped_brief(tmp_path: Path, **overrides) -> Path:
    fields = {"exa_target_firms": FIRMS, SCOPE_FIELD: SCOPE}
    fields.update(overrides)
    return _write_brief(tmp_path, **fields)


def _scope(tmp_path: Path, **overrides) -> CurrentEmployerScope:
    scope = CurrentEmployerScope.from_brief(load_brief(str(_scoped_brief(tmp_path, **overrides))))
    assert scope is not None
    return scope


def _candidate(**overrides) -> ExaCandidate:
    payload = {
        "identity_key": "linkedin.com/in/x",
        "name": "Sam Ro",
        "linkedin_url": "https://www.linkedin.com/in/x",
    }
    payload.update(overrides)
    return ExaCandidate(**payload)


def _with_render(markdown: str, **overrides) -> ExaCandidate:
    evidence = ExaEvidence(url="https://www.linkedin.com/in/x", text_excerpt=markdown)
    return _candidate(evidence=[evidence], **overrides)


# ---------------------------------------------------------------------------
# The rule, unit level
# ---------------------------------------------------------------------------


def test_a_brief_without_the_block_has_no_scope(tmp_path: Path) -> None:
    brief = load_brief(str(_write_brief(tmp_path, exa_target_firms=FIRMS)))
    assert CurrentEmployerScope.from_brief(brief) is None
    assert scope_errors(brief) == []


def test_target_firms_their_aliases_and_scope_aliases_are_in_scope(tmp_path: Path) -> None:
    scope = _scope(tmp_path)

    for firm, parent in (
        ("Benchmark", "Benchmark"),
        ("Benchmark Capital", "Benchmark"),
        ("Greylock", "Greylock"),
        ("Outlier", "Benchmark"),
        ("remotasks", "Benchmark"),
    ):
        verdict = scope.check(_candidate(current_title="Program Manager", current_firm=firm))
        assert (verdict.status, verdict.parent) == (IN_SCOPE, parent), firm


def test_an_identified_employer_outside_scope_is_named(tmp_path: Path) -> None:
    verdict = _scope(tmp_path).check(
        _candidate(current_title="Human Data Lead", current_firm="Acme Labs")
    )
    assert verdict.status == OUT_OF_SCOPE
    assert verdict.employer == "Acme Labs"
    assert verdict.note == "scope: current employer Acme Labs"


def test_no_identifiable_employer_is_unidentified(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    assert scope.check(_candidate()).status == UNIDENTIFIED
    # A headline with no "at X" names nobody.
    assert scope.check(_candidate(current_title="Program Manager")).status == UNIDENTIFIED


def test_a_headline_naming_a_competitor_is_in_scope(tmp_path: Path) -> None:
    verdict = _scope(tmp_path).check(_candidate(current_title="Coding expert at Outlier"))
    assert (verdict.status, verdict.parent) == (IN_SCOPE, "Benchmark")


def test_any_open_seat_at_a_competitor_is_in_scope(tmp_path: Path) -> None:
    candidate = _with_render(
        "# Sam Ro\nAdvisor\n500 connections\n\n## Experience\n"
        "### Advisor - [Acme](https://x) (Current)\nJan 2024 - Present (1 year)\n"
        "### Reviewer - [Outlier](https://y) (Current)\nJan 2025 - Present (6 months)\n",
        current_title="Advisor",
        current_firm="Acme",
    )
    assert _scope(tmp_path).check(candidate).status == IN_SCOPE


def test_a_past_competitor_seat_does_not_put_a_current_outsider_in_scope(
    tmp_path: Path,
) -> None:
    candidate = _with_render(
        "# Sam Ro\nHuman Data Lead\n500 connections\n\n## Experience\n"
        "### Human Data Lead - [Frontier Lab](https://x) (Current)\nJan 2025 - Present (1 year)\n"
        "### Program Manager - [Greylock](https://y)\nJan 2021 - Dec 2024 (4 years)\n",
    )
    verdict = _scope(tmp_path).check(candidate)
    assert verdict.status == OUT_OF_SCOPE
    assert verdict.employer == "Frontier Lab"


# ---------------------------------------------------------------------------
# Validation and lint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block, expected",
    [
        ("yes", "must be an object"),
        ({"alias": {}}, "unknown key 'alias'"),
        ({"aliases": ["Outlier"]}, "aliases must be an object"),
        ({"aliases": {"Outlier": ""}}, "aliases must map non-empty names"),
        ({"aliases": {"Outlier": "Scale AI"}}, "maps to 'Scale AI', which is not a target firm"),
        ({"aliases": {"Benchmark Capital": "Benchmark"}}, "already a query term"),
        ({"aliases": {"greylock": "Benchmark"}}, "already a query term"),
    ],
)
def test_a_malformed_block_is_a_blocking_lint_error(tmp_path: Path, block, expected) -> None:
    brief = load_brief(str(_scoped_brief(tmp_path, **{SCOPE_FIELD: block})))
    errors = scope_errors(brief)
    assert any(expected in error for error in errors), errors

    findings = lint_exa_brief(brief)
    blocking = blocking_findings(findings)
    assert [f.code for f in blocking] == ["employer_scope_invalid"]
    with pytest.raises(ValueError):
        CurrentEmployerScope.from_brief(brief)


def test_a_scope_with_no_target_firms_is_an_error(tmp_path: Path) -> None:
    brief = load_brief(str(_write_brief(tmp_path, exa_target_firms=[], **{SCOPE_FIELD: {}})))
    assert any("no target firms" in error for error in scope_errors(brief))


def test_scope_aliases_never_reach_a_query(tmp_path: Path) -> None:
    brief = load_brief(str(_scoped_brief(tmp_path)))
    assert lint_exa_brief(brief) == [] or not blocking_findings(lint_exa_brief(brief))
    for query in compile_exa_queries(brief):
        assert "Outlier" not in query.query and "Remotasks" not in query.query


# ---------------------------------------------------------------------------
# The pipeline: live verdicts, replay, and the slate
# ---------------------------------------------------------------------------


_RESULTS = {
    "Benchmark": [
        _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
        _person("patkim", "Pat Kim", "Human Data Lead", "Acme Labs"),
        _person("lenaoh", "Lena Oh", "Coding Reviewer", "Outlier"),
        FakeResult(
            url="https://www.linkedin.com/in/noemployer",
            title="Nia Vale | LinkedIn",
            text="Nia Vale\nDelivery lead.",
            highlights=["Nia Vale runs delivery programs."],
        ),
    ]
}
_SAVES = {"Avery Holt": "SAVE", "Pat Kim": "SAVE", "Lena Oh": "SAVE", "Nia Vale": "SAVE"}


def _section(slate: str, heading: str) -> str:
    return slate.split(heading, 1)[1].split("\n## ", 1)[0]


def test_an_out_of_scope_save_moves_to_review_and_is_never_dropped(
    tmp_path: Path, capsys
) -> None:
    judge = ScriptedJudge(dict(_SAVES))
    pipeline, _ = _pipeline(
        tmp_path, brief_path=_scoped_brief(tmp_path), results=_RESULTS, judge=judge
    )

    stats = asyncio.run(pipeline.run())

    # Judged first: the check never drops anyone before judgment.
    assert sorted(judge.seen) == ["Avery Holt", "Lena Oh", "Nia Vale", "Pat Kim"]
    assert stats.saved == 3 and stats.review == 1
    assert stats.scope_moved == 1

    # The stored verdict is the judge's; only the bucket moved.
    terminal = {
        row["display_name"]: row["terminal_decision"]
        for row in _rows(tmp_path, "SELECT display_name, terminal_decision FROM candidates")
    }
    assert terminal["Pat Kim"] == "SAVE"
    saved = {row["name"] for row in _jsonl(_state_dir(tmp_path) / "saves.jsonl")}
    assert saved == {"Avery Holt", "Lena Oh", "Nia Vale"}
    assert len(_rows(tmp_path, "SELECT * FROM workspace_entries")) == 3

    rows = {row["name"]: row for row in _jsonl(_state_dir(tmp_path) / "judgments.jsonl")}
    assert rows["Pat Kim"]["bucket"] == "review"
    assert rows["Pat Kim"]["decision"] == "SAVE"
    assert rows["Pat Kim"]["scope"] == "scope: current employer Acme Labs"
    assert rows["Avery Holt"]["scope"] == ""
    assert rows["Lena Oh"]["bucket"] == "save"

    out = capsys.readouterr().out.splitlines()
    pat = [line for line in out if "Pat Kim" in line]
    assert len(pat) == 1
    assert pat[0].startswith("    [exa] REVIEW Pat Kim")
    assert pat[0].endswith("· scope: current employer Acme Labs")

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    short_list = _section(slate, "## The short list")
    assert "Pat Kim" not in short_list
    for name in ("Avery Holt", "Lena Oh", "Nia Vale"):
        assert name in short_list, name
    worth = _section(slate, "## Worth a look")
    assert "**Pat Kim**" in worth
    assert "scope: current employer Acme Labs" in worth


def test_a_brief_without_the_block_leaves_every_save_alone(tmp_path: Path) -> None:
    judge = ScriptedJudge(dict(_SAVES))
    pipeline, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, exa_target_firms=FIRMS),
        results=_RESULTS,
        judge=judge,
    )

    stats = asyncio.run(pipeline.run())

    assert stats.saved == 4 and stats.review == 0 and stats.scope_moved == 0
    rows = {row["name"]: row for row in _jsonl(_state_dir(tmp_path) / "judgments.jsonl")}
    assert rows["Pat Kim"]["bucket"] == "save"
    assert rows["Pat Kim"]["scope"] == ""
    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    assert "Pat Kim" in _section(slate, "## The short list")


def test_the_slate_moves_a_stored_save_once_the_brief_opts_in(tmp_path: Path) -> None:
    first, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, exa_target_firms=FIRMS),
        results=_RESULTS,
        judge=ScriptedJudge(dict(_SAVES)),
    )
    asyncio.run(first.run())

    second_judge = ScriptedJudge({})
    second, _ = _pipeline(
        tmp_path, brief_path=_scoped_brief(tmp_path), results=_RESULTS, judge=second_judge
    )
    asyncio.run(second.run(resume=True))
    assert second_judge.seen == []

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    assert "Pat Kim" not in _section(slate, "## The short list")
    assert "scope: current employer Acme Labs" in _section(slate, "## Worth a look")


def test_an_interrupted_out_of_scope_save_is_not_replayed(tmp_path: Path, capsys) -> None:
    results = {"Benchmark": [_person("patkim", "Pat Kim", "Human Data Lead", "Acme Labs")]}
    first, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, exa_target_firms=FIRMS),
        results=results,
        judge=ScriptedJudge({"Pat Kim": "SAVE"}),
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

    second, _ = _pipeline(
        tmp_path,
        brief_path=_scoped_brief(tmp_path),
        results=results,
        judge=ScriptedJudge({}),
    )
    stats = asyncio.run(second.run(resume=True))

    assert stats.replayed_saves == 0
    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []
    out = capsys.readouterr().out
    assert (
        "    [exa] Pat Kim — interrupted save not replayed: scope: current employer Acme Labs"
        in out.splitlines()
    )


def test_a_fault_in_the_scope_check_leaves_the_verdict_alone(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import exa.scope_gate as scope_gate

    def boom(*_args, **_kwargs):
        raise RuntimeError("scope matcher exploded")

    monkeypatch.setattr(scope_gate, "match_current_employer", boom)

    pipeline, _ = _pipeline(
        tmp_path,
        brief_path=_scoped_brief(tmp_path),
        results={"Benchmark": [_person("patkim", "Pat Kim", "Human Data Lead", "Acme Labs")]},
        judge=ScriptedJudge({"Pat Kim": "SAVE"}),
    )
    stats = asyncio.run(pipeline.run())

    assert stats.saved == 1 and stats.scope_moved == 0
    warnings = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "scope check failed" in line
    ]
    assert warnings and all(line.startswith("    [exa] ") for line in warnings)


def test_a_scope_alias_written_into_an_archetype_query_blocks_the_launch(
    tmp_path: Path,
) -> None:
    brief = load_brief(
        str(
            _scoped_brief(
                tmp_path,
                exa_query_lanes=["people_search", "archetype"],
                exa_archetype_queries=[
                    {
                        "id": "outlier-reviewers",
                        "text": "Coding reviewers at Outlier in the SF Bay Area",
                        "num_results": 25,
                    }
                ],
            )
        )
    )
    blocking = blocking_findings(lint_exa_brief(brief))
    assert [f.code for f in blocking] == ["employer_scope_invalid"]
    assert "'Outlier' appears in a compiled query" in blocking[0].message


def test_check_says_the_scope_check_is_on(tmp_path: Path, monkeypatch, capsys) -> None:
    import exa.health as health
    from exa import run as exa_run

    class _Ready:
        ready = True
        blockers: list = []

    monkeypatch.setattr(health, "probe_exa_readiness", lambda: _Ready())
    brief = _scoped_brief(tmp_path)
    code = exa_run.main(
        ["--check", "--brief", str(brief), "--state-dir", str(tmp_path / "state")]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert (
        "    [exa] current-employer scope check on: 5 names across 2 firms; "
        "an out-of-scope save goes to review"
    ) in out.splitlines()

    capsys.readouterr()
    exa_run.main(
        [
            "--check",
            "--brief",
            str(_write_brief(tmp_path, exa_target_firms=FIRMS)),
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )
    assert "scope check on" not in capsys.readouterr().out
