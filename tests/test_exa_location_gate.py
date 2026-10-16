"""Exa location scope: a card location outside the brief's countries never reaches the judge.

The Exa path never read location before judgment, and 22 of 79 Exa saves on
the SPL competitor run (req 1238) sat outside the United States. The judge
misread the record too: it called Idris Karim's location "a single
web-derived United Kingdom tag" when his card says Edmonton, Alberta, and it
took a company's headquarters line on a role (Pareto's "Stanford, CA" on
Otto Ngugi's seats, whose card says Nairobi) for the person's residence.

These tests pin the deterministic gate the brief opts into with
``exa_location_scope``:

* the person's CARD location is read, never a role's location line;
* only a location POSITIVELY outside the allowed countries is excluded;
  an empty or unresolvable one goes to the judge;
* an excluded candidate is a tagged, counted REJECT with no judge call, like
  the blacklist gates, and a stored save outside the scope leaves the slate;
* a brief without the field behaves exactly as before.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from exa.lane_compiler import blocking_findings, lint_exa_brief
from exa.location_gate import (
    INSIDE,
    LOCATION_GATE_PATH,
    OUTSIDE,
    UNKNOWN,
    LocationScope,
    location_scope_errors,
)
from exa.schemas import ExaCandidate
from tests.test_exa_orchestrator import (
    FakeResult,
    ScriptedJudge,
    _brief_dict,
    _jsonl,
    _pipeline,
    _state_dir,
    _store,
    _write_brief,
)


US_ONLY = {"countries": ["United States"]}


def _render(location: str, experiences: list[dict] | None = None) -> dict:
    return {
        "name": "Test Person",
        "headline": "Strategic Project Lead",
        "location": location,
        "about": "",
        "total_experience_text": "",
        "total_experience_years": None,
        "experiences": experiences or [],
        "education": [],
    }


def _candidate(location: str = "", *, render_location: str | None = None, roles=None):
    return ExaCandidate(
        identity_key="linkedin.com/in/test",
        name="Test Person",
        current_title="Strategic Project Lead",
        current_firm="Pareto AI",
        location=location,
        linkedin_url="https://www.linkedin.com/in/test",
        profile_render=(
            _render(render_location, roles) if render_location is not None else None
        ),
    )


def _scope(**block) -> LocationScope:
    return LocationScope.from_brief({"exa_location_scope": block or US_ONLY})


# ---------------------------------------------------------------------------
# Resolution: the card, positively
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "card, expected",
    [
        # The audit's misreads, as their cards actually read.
        ("Edmonton, Alberta, Canada (CA)", OUTSIDE),
        ("Toronto, Ontario, Canada (CA)", OUTSIDE),
        ("Nairobi County, Kenya (KE)", OUTSIDE),
        ("Hyderabad, Telangana, India (IN)", OUTSIDE),
        ("Bangladesh (BD)", OUTSIDE),
        ("Canada (CA)", OUTSIDE),
        # In scope.
        ("United States (US)", INSIDE),
        ("San Francisco Bay Area (US)", INSIDE),
        ("Colorado Springs, Colorado, United States (US)", INSIDE),
        ("San Juan, Puerto Rico (PR)", INSIDE),
    ],
)
def test_card_country_code_decides(card: str, expected: str) -> None:
    assert _scope().check(_candidate(render_location=card)).status == expected


@pytest.mark.parametrize(
    "card, expected",
    [
        ("Hyderabad, Telangana, India", OUTSIDE),
        ("Dhaka, Bangladesh", OUTSIDE),
        ("Belgrade, Serbia", OUTSIDE),
        ("Portland, Oregon, United States", INSIDE),
        ("Austin, Texas", INSIDE),
        ("United States", INSIDE),
        # Not resolvable to a country: the judge decides.
        ("Greater Seattle Area", UNKNOWN),
        ("Bay Area", UNKNOWN),
        ("Tbilisi, Georgia", UNKNOWN),  # a state and a country both
        ("Brookline, MA", UNKNOWN),
        ("", UNKNOWN),
    ],
)
def test_card_text_without_a_code_resolves_only_when_unambiguous(
    card: str, expected: str
) -> None:
    assert _scope().check(_candidate(render_location=card)).status == expected


def test_a_role_location_never_overrides_the_card() -> None:
    """Otto Ngugi: Pareto's HQ on every role, Nairobi on the card."""

    roles = [
        {"title": "Strategic Project Lead", "company": "Pareto AI",
         "location": "Stanford, California, United States", "is_current": True},
    ]
    verdict = _scope().check(
        _candidate(render_location="Nairobi County, Kenya (KE)", roles=roles)
    )
    assert verdict.status == OUTSIDE
    assert verdict.location == "Nairobi County, Kenya"
    assert verdict.country == "KE"


def test_a_role_location_alone_is_not_a_person_location() -> None:
    """No card location: a role's San Francisco line does not place anyone."""

    roles = [
        {"title": "Strategic Project Lead", "company": "Mercor",
         "location": "Bengaluru, Karnataka, India", "is_current": True},
    ]
    assert _scope().check(_candidate(render_location="", roles=roles)).status == UNKNOWN


def test_record_location_is_read_when_there_is_no_render() -> None:
    assert _scope().check(_candidate("Gurugram, Haryana, India")).status == OUTSIDE
    assert _scope().check(_candidate("San Francisco, California")).status == INSIDE
    assert _scope().check(_candidate("")).status == UNKNOWN


def test_scope_accepts_codes_and_several_countries() -> None:
    scope = _scope(countries=["US", "Canada"])
    assert scope.check(_candidate(render_location="Toronto, Ontario, Canada (CA)")).status == INSIDE
    assert scope.check(_candidate(render_location="London, England, United Kingdom (GB)")).status == OUTSIDE


# ---------------------------------------------------------------------------
# The brief field: opt-in, validated
# ---------------------------------------------------------------------------


def test_scope_is_none_without_the_field() -> None:
    assert LocationScope.from_brief(_brief_dict()) is None
    assert location_scope_errors(_brief_dict()) == []


@pytest.mark.parametrize(
    "block, fragment",
    [
        ("United States", "must be an object"),
        ({}, "countries"),
        ({"countries": []}, "countries"),
        ({"countries": ["Atlantis"]}, "Atlantis"),
        ({"countries": ["United States"], "cities": ["SF"]}, "unknown key"),
    ],
)
def test_malformed_scope_is_a_blocking_lint_finding(block, fragment) -> None:
    brief = _brief_dict(exa_location_scope=block)
    errors = location_scope_errors(brief)
    assert errors and any(fragment in error for error in errors)
    findings = blocking_findings(lint_exa_brief(brief))
    assert any(f.code == "location_scope_invalid" for f in findings)


def test_valid_scope_lints_clean() -> None:
    findings = lint_exa_brief(_brief_dict(exa_location_scope=US_ONLY))
    assert not [f for f in findings if f.code == "location_scope_invalid"]


# ---------------------------------------------------------------------------
# The pipeline: gated before judgment, tagged, counted
# ---------------------------------------------------------------------------


def _located(handle: str, name: str, firm: str, location: str) -> FakeResult:
    return FakeResult(
        url=f"https://www.linkedin.com/in/{handle}",
        title=f"{name} - Strategic Project Lead - {firm} | LinkedIn",
        text=f"{location}\n{name} works at {firm}.",
        highlights=[f"{name} is a Strategic Project Lead at {firm}."],
    )


_MIXED = {
    "Benchmark": [
        _located("nikovarga", "Niko Varga", "Benchmark", "Hyderabad, Telangana, India"),
        _located("ezrapell", "Ezra Pell", "Benchmark", "San Francisco, California"),
        _located("nolocation", "Nora Lee", "Benchmark", "Senior Lead"),
    ]
}


def test_outside_the_scope_is_rejected_before_judgment(tmp_path: Path, capsys) -> None:
    brief_path = _write_brief(tmp_path, exa_location_scope=US_ONLY)
    judge = ScriptedJudge({"Niko Varga": "SAVE", "Ezra Pell": "SAVE", "Nora Lee": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, brief_path=brief_path, results=_MIXED, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Ezra Pell", "Nora Lee"], "unknown location still goes to the judge"
    assert stats.location_gated == 1
    assert stats.judged == 2

    row = _store(tmp_path).get_candidate(
        source="exa", brief_id=pipeline.brief_id, identity_key="linkedin.com/in/nikovarga"
    )
    assert row["terminal_decision"] == "REJECT"
    payload = json.loads(row["terminal_payload_json"])
    assert payload["full_decision"]["path"] == LOCATION_GATE_PATH
    assert payload["full_decision"]["reject_reason"] == "HARD_GATE"
    assert payload["gate"] == LOCATION_GATE_PATH

    rows = {r["name"]: r for r in _jsonl(_state_dir(tmp_path) / "judgments.jsonl")}
    assert rows["Niko Varga"]["gate"] == LOCATION_GATE_PATH
    assert rows["Niko Varga"]["rationale"] == (
        "located outside the United States (Hyderabad, Telangana, India)"
    )
    assert rows["Ezra Pell"]["gate"] == ""

    out = capsys.readouterr().out
    assert (
        "    [exa] SKIP   Niko Varga — Strategic Project Lead, Benchmark "
        "· outside the United States (Hyderabad, Telangana, India)"
    ) in out.splitlines()

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    assert "## Excluded: outside the United States — 1" in slate
    assert "1 person rejected before judgment" in slate.split(
        "## Excluded: outside the United States", 1
    )[1]
    report = json.loads((_state_dir(tmp_path) / "run-report.json").read_text())
    assert report["metrics_summary"]["gated_location_scope"] == 1
    # Bucketed like any other REJECT, exactly as the blacklist gates are
    # (exa/report.py RunView.gated_employer).
    assert report["metrics_summary"]["rejected"] == 1


def test_brief_without_the_field_is_inert(tmp_path: Path) -> None:
    judge = ScriptedJudge({"Niko Varga": "SAVE", "Ezra Pell": "SAVE", "Nora Lee": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, results=_MIXED, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Niko Varga", "Ezra Pell", "Nora Lee"]
    assert stats.location_gated == 0
    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    assert "## Excluded" not in slate


def test_slate_moves_a_stored_save_outside_the_scope(tmp_path: Path) -> None:
    """A save judged before the brief opted in leaves the short list, unre-judged."""

    first_judge = ScriptedJudge({"Niko Varga": "SAVE", "Ezra Pell": "SAVE"})
    first, _ = _pipeline(
        tmp_path, brief_path=_write_brief(tmp_path), results=_MIXED, judge=first_judge
    )
    asyncio.run(first.run())

    second_judge = ScriptedJudge({})
    second, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, exa_location_scope=US_ONLY),
        results=_MIXED,
        judge=second_judge,
    )
    asyncio.run(second.run(resume=True))
    assert second_judge.seen == []

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    short_list = slate.split("## The short list", 1)[1].split("\n## ", 1)[0]
    assert "Ezra Pell" in short_list
    assert "Niko Varga" not in short_list
    excluded = slate.split("## Excluded: outside the United States — 1", 1)[1].split("\n## ", 1)[0]
    assert "**Niko Varga**" in excluded
    assert "Hyderabad, Telangana, India" in excluded
    report = json.loads((_state_dir(tmp_path) / "run-report.json").read_text())
    assert report["metrics_summary"]["excluded_location_scope"] == 1
    assert report["metrics_summary"]["saved"] == 1


def test_gate_fault_falls_through_to_judgment(tmp_path: Path, monkeypatch, capsys) -> None:
    import exa.location_gate as location_gate

    def _boom(self, candidate):
        raise RuntimeError("resolver broke")

    monkeypatch.setattr(location_gate.LocationScope, "check", _boom)
    brief_path = _write_brief(tmp_path, exa_location_scope=US_ONLY)
    judge = ScriptedJudge({})
    pipeline, _ = _pipeline(tmp_path, brief_path=brief_path, results=_MIXED, judge=judge)

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Niko Varga", "Ezra Pell", "Nora Lee"]
    assert stats.location_gated == 0
    out = capsys.readouterr().out
    assert sum("location check failed" in line for line in out.splitlines()) == 3


def test_interrupted_save_outside_the_scope_is_not_replayed(
    tmp_path: Path, capsys
) -> None:
    """A save stranded before its workspace write never lands once the scope rules it out."""

    from shared.save_destination.candidate_workspace import (
        CandidateWorkspaceSaveDestination,
    )

    only_niko = {"Benchmark": _MIXED["Benchmark"][:1]}
    first, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path),
        results=only_niko,
        judge=ScriptedJudge({"Niko Varga": "SAVE"}),
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
    capsys.readouterr()

    second, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, exa_location_scope=US_ONLY),
        results=only_niko,
        judge=ScriptedJudge({}),
    )
    stats = asyncio.run(second.run(resume=True))

    assert stats.replayed_saves == 0
    assert "interrupted save not replayed: outside the United States" in capsys.readouterr().out
    from tests.test_exa_orchestrator import _rows

    assert _rows(tmp_path, "SELECT * FROM workspace_entries") == []
