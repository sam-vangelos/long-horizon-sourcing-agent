"""Exa employer-blacklist gate: current employees never reach the judge.

The blacklist used to reach an Exa judgment only as prompt text, and a
model that honors a rule sixteen times can still miss it on the
seventeenth. These tests pin the deterministic gate in front of the judge
and the same rule applied when the slate is rendered:

* a candidate whose CURRENT employer (the record's firm, an open position,
  or a headline "at X" / "@ X") is on the blacklist is rejected with no
  judge call and a machine-readable marker;
* a past employer never gates; alumni stay in the pool;
* a stored save at a blacklisted current employer leaves the short list
  on the next render, without a re-judgment;
* a fault inside the gate falls through to ordinary judgment.

The pipeline tests run against the real store, through the fixtures in
``tests/test_exa_orchestrator.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from exa.schemas import ExaCandidate, ExaEvidence
from tests.test_exa_orchestrator import (
    FakeResult,
    ScriptedJudge,
    _jsonl,
    _person,
    _pipeline,
    _rows,
    _state_dir,
    _store,
    _write_brief,
)


BLACKLIST = ["Centific", "Mercor"]


def _render_person(handle: str, name: str, page_title: str, markdown: str) -> FakeResult:
    """A people-search hit whose text is Exa's LinkedIn markdown render."""

    return FakeResult(
        url=f"https://www.linkedin.com/in/{handle}",
        title=page_title,
        text=markdown,
        highlights=[f"{name} profile."],
    )


def _headline_only(handle: str, name: str, headline: str) -> FakeResult:
    """A render with a headline and no Experience section: no firm, no title."""

    return _render_person(
        handle,
        name,
        f"{name} | LinkedIn",
        f"# {name}\n{headline}\nSan Francisco, California, United States\n"
        "500 connections\n",
    )


def _terminal(tmp_path: Path, pipeline, handle: str) -> dict:
    row = _store(tmp_path).get_candidate(
        source="exa",
        brief_id=pipeline.brief_id,
        identity_key=f"linkedin.com/in/{handle}",
    )
    assert row is not None, f"{handle} was never persisted"
    return dict(row)


# ---------------------------------------------------------------------------
# (a) current employer on the blacklist: gated, no judge call
# ---------------------------------------------------------------------------


def test_current_employer_on_blacklist_is_rejected_without_a_judge_call(
    tmp_path: Path, capsys
) -> None:
    brief_path = _write_brief(tmp_path, employer_blacklist=BLACKLIST)
    judge = ScriptedJudge({"Ezra Tam": "SAVE", "Avery Holt": "SAVE"})
    results = {
        "Benchmark": [
            _person("ezratam", "Ezra Tam", "Technical Program Manager", "Centific"),
            _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
        ]
    }
    pipeline, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=results, judge=judge
    )

    stats = asyncio.run(pipeline.run())

    assert "Ezra Tam" not in judge.seen, "a blacklisted current employee was judged"
    assert judge.seen == ["Avery Holt"]

    ezra = _terminal(tmp_path, pipeline, "ezratam")
    assert ezra["current_lifecycle_state"] == "full_terminal"
    assert ezra["terminal_decision"] == "REJECT"
    payload = json.loads(ezra["terminal_payload_json"])
    assert payload["full_decision"]["path"] == "employer_blacklist"
    assert payload["full_decision"]["reject_reason"] == "HARD_GATE"

    rows = {row["name"]: row for row in _jsonl(_state_dir(tmp_path) / "judgments.jsonl")}
    assert rows["Ezra Tam"]["decision"] == "REJECT"
    assert rows["Ezra Tam"]["gate"] == "employer_blacklist"
    assert (
        rows["Ezra Tam"]["rationale"]
        == "current employer on the employer blacklist (Centific)"
    )
    assert rows["Avery Holt"]["gate"] == ""

    out = capsys.readouterr().out
    assert (
        "    [exa] SKIP   Ezra Tam — Technical Program Manager, Centific "
        "· employer blacklist (Centific)"
    ) in out.splitlines()
    assert stats.blacklist_gated == 1
    assert stats.judged == 1, "a gated candidate is not a judgment"
    assert stats.saved == 1

    workspace = _rows(tmp_path, "SELECT * FROM workspace_entries")
    assert len(workspace) == 1, "only the judged save reached the workspace"


# ---------------------------------------------------------------------------
# (b) past employer only: judged normally
# ---------------------------------------------------------------------------


_ALUMNUS_RENDER = """# Pat Kim
Senior Engineer at Acme Robotics
San Francisco, California, United States
500 connections

## Experience
### Senior Engineer - [Acme Robotics](https://www.linkedin.com/company/acme) (Current)
Jan 2023 - Present (2 years)
### Data Engineer - [Centific](https://www.linkedin.com/company/centific)
Jan 2020 - Dec 2022 (3 years)
"""


def test_past_employer_on_blacklist_is_judged_normally(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, employer_blacklist=BLACKLIST)
    judge = ScriptedJudge({"Pat Kim": "SAVE"})
    results = {
        "Benchmark": [
            _render_person(
                "patkim",
                "Pat Kim",
                "Pat Kim - Senior Engineer - Acme Robotics | LinkedIn",
                _ALUMNUS_RENDER,
            )
        ]
    }
    pipeline, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=results, judge=judge
    )

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Pat Kim"]
    assert _terminal(tmp_path, pipeline, "patkim")["terminal_decision"] == "SAVE"
    assert stats.blacklist_gated == 0


# ---------------------------------------------------------------------------
# (c) headline employer context gates; a bare mention does not
# ---------------------------------------------------------------------------


def test_headline_at_employer_gates_and_bare_mentions_do_not(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, employer_blacklist=BLACKLIST)
    judge = ScriptedJudge({})
    results = {
        "Benchmark": [
            _headline_only("danalee", "Dana Lee", "Head of Data at Mercor"),
            _headline_only("elipark", "Eli Park", "Data Scientist | ex-Mercor"),
            _headline_only("faywu", "Fay Wu", "Mercor alumni | ML Engineer"),
        ]
    }
    pipeline, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=results, judge=judge
    )

    asyncio.run(pipeline.run())

    assert "Dana Lee" not in judge.seen, "a headline 'at Mercor' was judged"
    assert sorted(judge.seen) == ["Eli Park", "Fay Wu"]
    dana = _terminal(tmp_path, pipeline, "danalee")
    assert dana["terminal_decision"] == "REJECT"
    payload = json.loads(dana["terminal_payload_json"])
    assert payload["full_decision"]["path"] == "employer_blacklist"


# ---------------------------------------------------------------------------
# (d) a fault inside the matcher falls through to judgment
# ---------------------------------------------------------------------------


def test_gate_fault_falls_through_to_judgment(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import exa.employer_gate as employer_gate

    def boom(*_args, **_kwargs):
        raise RuntimeError("matcher exploded")

    monkeypatch.setattr(employer_gate, "match_current_employer", boom)

    brief_path = _write_brief(tmp_path, employer_blacklist=BLACKLIST)
    judge = ScriptedJudge({"Ezra Tam": "REJECT"})
    results = {
        "Benchmark": [
            _person("ezratam", "Ezra Tam", "Technical Program Manager", "Centific")
        ]
    }
    pipeline, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=results, judge=judge
    )

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Ezra Tam"], "the judge still ran after the gate faulted"
    assert stats.blacklist_gated == 0
    assert stats.judged == 1
    out = capsys.readouterr().out
    warnings = [line for line in out.splitlines() if "employer blacklist check failed" in line]
    assert len(warnings) == 1
    assert warnings[0].startswith("    [exa] ")


# ---------------------------------------------------------------------------
# (e) the slate excludes a stored save at a blacklisted current employer
# ---------------------------------------------------------------------------


def test_slate_moves_a_stored_blacklisted_save_out_of_the_short_list(
    tmp_path: Path,
) -> None:
    results = {
        "Benchmark": [
            _person("ezratam", "Ezra Tam", "Technical Program Manager", "Centific"),
            _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
        ]
    }
    # Session 1 stores both saves: the brief does not carry the blacklist
    # yet, which is the same stored state a judge miss leaves behind.
    first_judge = ScriptedJudge({"Ezra Tam": "SAVE", "Avery Holt": "SAVE"})
    first, _ = _pipeline(
        tmp_path, brief_path=_write_brief(tmp_path), results=results, judge=first_judge
    )
    asyncio.run(first.run())
    assert sorted(first_judge.seen) == ["Avery Holt", "Ezra Tam"]

    # Session 2 resumes under the blacklist. No query is left to run and
    # nobody is re-judged; the slate is rendered from stored verdicts.
    brief_path = _write_brief(tmp_path, employer_blacklist=BLACKLIST)
    second_judge = ScriptedJudge({})
    second, _ = _pipeline(
        tmp_path, brief_path=brief_path, results=results, judge=second_judge
    )
    asyncio.run(second.run(resume=True))
    assert second_judge.seen == []

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    short_list = slate.split("## The short list", 1)[1].split("\n## ", 1)[0]
    assert "Avery Holt" in short_list
    assert "Ezra Tam" not in short_list, "a blacklisted current employee stayed on the short list"
    assert "## Excluded: current employer on the blacklist — 1" in slate
    excluded = slate.split("## Excluded: current employer on the blacklist", 1)[1]
    excluded = excluded.split("\n## ", 1)[0]
    assert "**Ezra Tam**" in excluded
    assert "Centific" in excluded
    verify = slate.split("## Verify before outreach", 1)[1]
    assert "Ezra Tam" not in verify

    report = json.loads((_state_dir(tmp_path) / "run-report.json").read_text())
    assert report["metrics_summary"]["saved"] == 1
    assert report["metrics_summary"]["excluded_employer_blacklist"] == 1


def test_slate_counts_gated_rejections_in_the_excluded_section(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, employer_blacklist=BLACKLIST)
    results = {
        "Benchmark": [
            _person("ezratam", "Ezra Tam", "Technical Program Manager", "Centific"),
            _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
        ]
    }
    pipeline, _ = _pipeline(
        tmp_path,
        brief_path=brief_path,
        results=results,
        judge=ScriptedJudge({"Avery Holt": "SAVE"}),
    )
    asyncio.run(pipeline.run())

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    assert "## Excluded: current employer on the blacklist — 1" in slate
    assert "1 person rejected before judgment" in slate


def test_slate_without_exclusions_has_no_excluded_section(tmp_path: Path) -> None:
    pipeline, _ = _pipeline(
        tmp_path,
        results={
            "Benchmark": [_person("averylinholt", "Avery Holt", "Investor", "Benchmark")]
        },
        judge=ScriptedJudge({"Avery Holt": "SAVE"}),
    )
    asyncio.run(pipeline.run())

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    assert "## Excluded" not in slate


# ---------------------------------------------------------------------------
# The current-employer signals, unit level
# ---------------------------------------------------------------------------


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


def test_any_open_position_at_a_blacklisted_employer_gates() -> None:
    from exa.employer_gate import employer_blacklist_match

    candidate = _with_render(
        "# Sam Ro\nAdvisor\n500 connections\n\n## Experience\n"
        "### Advisor - [Acme](https://x) (Current)\nJan 2024 - Present (1 year)\n"
        "### Annotator - [Mercor](https://y) (Current)\nJan 2025 - Present (6 months)\n",
        current_title="Advisor",
        current_firm="Acme",
    )

    hit = employer_blacklist_match(candidate, BLACKLIST)
    assert hit is not None and hit[0] == "Mercor" and hit[1] == "current_company"


def test_most_recent_position_stands_in_when_none_is_open() -> None:
    from exa.employer_gate import employer_blacklist_match

    candidate = _with_render(
        "# Sam Ro\n500 connections\n\n## Experience\n"
        "### Program Manager - [Centific](https://x)\nJan 2021 - Mar 2025 (4 years)\n"
        "### Analyst - [Acme](https://y)\nJan 2018 - Dec 2020 (3 years)\n",
    )

    hit = employer_blacklist_match(candidate, BLACKLIST)
    assert hit is not None and hit[0] == "Centific"


def test_title_naming_the_employer_gates_like_a_headline() -> None:
    from exa.employer_gate import employer_blacklist_match

    assert employer_blacklist_match(
        _candidate(current_title="TPM at Centific"), BLACKLIST
    ) is not None
    assert employer_blacklist_match(
        _candidate(current_title="TPM, formerly at Centific"), BLACKLIST
    ) is None


def test_empty_blacklist_and_clean_record_do_not_gate() -> None:
    from exa.employer_gate import employer_blacklist_match

    ezra = _candidate(current_title="TPM", current_firm="Centific")
    assert employer_blacklist_match(ezra, []) is None
    assert employer_blacklist_match(ezra, None) is None
    assert employer_blacklist_match(
        _candidate(current_title="Investor", current_firm="Benchmark"), BLACKLIST
    ) is None


@pytest.mark.parametrize(
    "company, headline, expected",
    [
        ("Anthropic", "", ("Anthropic", "current_company", "Anthropic")),
        ("", "MTS @ OpenAI", ("OpenAI", "headline", "MTS @ OpenAI")),
        ("", "ML Engineer | ex-OpenAI", None),
        ("Babbage School of Engineering", "", None),
        ("Centurion Systems", "", None),
    ],
)
def test_shared_matcher_keys_on_the_current_employer(company, headline, expected) -> None:
    from shared.employer_blacklist import match_current_employer

    assert (
        match_current_employer(company, headline, ["Babbage", "OpenAI", "Anthropic"])
        == expected
    )
