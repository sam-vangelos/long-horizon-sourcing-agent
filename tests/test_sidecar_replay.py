"""tools/sidecar_replay.py: a stored plan replayed through the sidecar logic.

The tool is read-only by contract, so every test that runs it also proves the
state directory is byte for byte what it was. No network: the sidecar is a
fake standing in for ``urlopen``, or a plain function standing in for the
client.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

import shared.config as config
from linkedin import boolean_enrichment
from tests.test_boolean_sidecar import MIXED, MIXED_RICHER, SWEEP, _FakeSidecar
from tools import sidecar_replay

ILLEGAL = '("evaluation" AND'
TWO_WHOLE_GROUPS = '("evaluation") AND ("RLHF")'
CODE_GEN = '("code generation" OR "code gen")'


def _plan() -> dict:
    return {
        "generated_strings": [
            {
                # A string the live sidecar already edited: replayed as authored.
                "boolean": MIXED_RICHER,
                "boolean_original": MIXED,
                "boolean_sidecar": {"adapter": "a", "requests": 3, "kept": []},
            },
            {"boolean": TWO_WHOLE_GROUPS},
            {"boolean": ILLEGAL},
            "not a string item",
        ],
        "coverage_gaps": [{"suggested_boolean": SWEEP}],
    }


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    (tmp_path / "execution_plan.json").write_text(json.dumps(_plan(), indent=2))
    (tmp_path / "progress.json").write_text("{}")
    return tmp_path


@pytest.fixture(autouse=True)
def _client_settings(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_URL", "http://configured.test")
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_SAMPLES", 2)
    # The flag stays off: the replay must not depend on it.
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_ENABLED", False)


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(directory.iterdir())
    }


def test_the_plan_is_replayed_as_authored_and_nothing_is_written(
    monkeypatch, capsys, state_dir
):
    before = _snapshot(state_dir)
    fake = _FakeSidecar({"register": [CODE_GEN]})
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    code = sidecar_replay.main([str(state_dir), "--url", "http://replay.test"])
    out = capsys.readouterr().out

    assert code == 0
    assert _snapshot(state_dir) == before
    assert {r["url"] for r in fake.requests} == {"http://replay.test/enrich"}
    givens = [r["given"] for r in fake.requests]
    # The original was sent, not the edited string.
    assert '("code generation")' in givens
    assert not any("code gen\"" in given for given in givens)
    assert "Scale AI" not in "".join(givens)
    assert len(fake.requests) == 9

    assert "generated_strings[0]  whole groups 1, concept runs 2 (2 sent), requests 3" in out
    assert "generated_strings[1]  whole groups 2, concept runs 0 (0 sent), requests 2" in out
    assert "generated_strings[2]  does not parse, so it is never sent" in out
    assert "coverage_gaps[0]  whole groups 0, concept runs 3 (3 sent), requests 4" in out
    assert '  kept register_run, group 0, code-generation + ["code gen"]: ' in out
    assert f"    after   {MIXED_RICHER}" in out
    assert "strings 4, eligible 3, changed 1, requests 9, " in out
    assert "sidecar http://replay.test" in out


def test_the_configured_address_is_the_default(monkeypatch, capsys, state_dir):
    fake = _FakeSidecar()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert sidecar_replay.main([str(state_dir), "--limit", "1"]) == 0
    assert {r["url"] for r in fake.requests} == {"http://configured.test/enrich"}
    assert "strings 1, eligible 1, changed 0, requests 3, " in capsys.readouterr().out


def test_a_dead_sidecar_stops_the_replay_and_says_so(monkeypatch, capsys, state_dir):
    before = _snapshot(state_dir)

    def _hang(request):
        raise TimeoutError("timed out")

    fake = _FakeSidecar(fail=_hang)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert sidecar_replay.main([str(state_dir)]) == 1
    out = capsys.readouterr().out
    assert "generated_strings[0]  the sidecar stopped answering" in out
    assert "generated_strings[1]" not in out
    assert len(fake.requests) == 2
    assert _snapshot(state_dir) == before


def test_a_missing_plan_is_refused(tmp_path, capsys):
    assert sidecar_replay.main([str(tmp_path)]) == 2
    assert "cannot read" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_originals_are_taken_on_both_surfaces():
    plan = {
        "generated_strings": [{"boolean": "b", "boolean_original": "a"}, {"boolean": "c"}],
        "coverage_gaps": [
            {"suggested_boolean": "e", "suggested_boolean_original": "d"},
            {"suggested_boolean": "  "},
        ],
    }
    assert list(sidecar_replay.plan_strings(plan)) == [
        ("generated_strings[0]", "a"),
        ("generated_strings[1]", "c"),
        ("coverage_gaps[0]", "d"),
    ]


def test_replay_with_a_fake_asker_reports_each_change_and_the_totals():
    lexicon = boolean_enrichment.load_role_lexicon()
    asked: list[tuple[str, str]] = []

    def ask(task: str, given: str) -> list[str]:
        asked.append((task, given))
        return [CODE_GEN] if given == '("code generation")' else []

    ticks = iter([0.0, 2.5, 10.0, 11.0])
    lines: list[str] = []
    totals = sidecar_replay.replay(
        [("s0", MIXED), ("s1", TWO_WHOLE_GROUPS)],
        lexicon,
        ask,
        emit=lines.append,
        clock=lambda: next(ticks),
    )
    assert totals == {
        "strings": 2,
        "eligible": 2,
        "changed": 1,
        "requests": 5,
        "seconds": 3.5,
        "stopped": False,
    }
    assert lines[0] == "s0  whole groups 1, concept runs 2 (2 sent), requests 3, 2.5 s"
    assert lines[2].startswith('  kept register_run, group 0, code-generation + ["code gen"]: ')
    assert lines[3] == f"    before  {MIXED}"
    assert lines[4] == f"    after   {MIXED_RICHER}"
    assert lines[-1] == "strings 2, eligible 2, changed 1, requests 5, 3.5 s"
    assert len(asked) == 5
