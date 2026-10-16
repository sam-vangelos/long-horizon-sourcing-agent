"""Registry contract for the Exa source (Exa module S6).

Mirrors :mod:`tests.test_cloris_launchers` (the exec_search contract
every source must satisfy) with the Exa-specific seams added:

- the argv shape ``exa/run.py``'s argparse actually accepts, INCLUDING
  the ``-u`` that makes Exa's argv prefix four elements rather than
  three (the in-process dispatcher slices on that);
- the state dir living under ``output/state/exa/``;
- the save-destination slot carrying Exa's brief-shaped launch gate —
  the first BLOCKING finding from ``exa.lane_compiler.lint_exa_brief``,
  and ``None`` for a brief that lints clean;
- the readiness probe being wired to ``exa.health.probe_exa_readiness``
  rather than re-implemented (monkeypatched, so no network, no key).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cloris.launchers import (
    LAUNCHERS,
    LauncherEntry,
    SaveDestinationBlocker,
    get_launcher,
    known_sources,
)
from shared.runtime_state.store import EXA_QUERY_KIND


def _write_brief(path: Path, **overrides) -> Path:
    """A minimal V2 brief that lints clean on the Exa surface."""

    payload = {
        "id": "exa-registry-fixture",
        "role_title": "Chief of Staff Fixture",
        "hiring_company": "Babbage",
        "employer_blacklist": ["Babbage", "OpenAI", "Anthropic"],
        "geography": "San Francisco Bay Area",
        "capability_areas": [{"name": "x", "description": "y"}],
        "depth_distinction": {
            "builder_definition": "a",
            "user_definition": "b",
            "edge_case_guidance": "c",
        },
        "exa_target_firms": [
            {"name": "Benchmark", "domain": "benchmark.com"},
            {"name": "Kiln Ventures", "domain": "kilnventures.com"},
        ],
        "exa_query_lanes": ["people_search", "team_page"],
        "exa_title_terms": ["Senior Associate", "Principal"],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))
    return path


def test_known_sources_includes_exa() -> None:
    sources = known_sources()
    assert "exa" in sources
    # Stable ascending order is the registry's contract (the API layer
    # renders it as a 422 ``allowed`` list).
    assert sources == tuple(sorted(sources))


def test_exa_launcher_entry_shape() -> None:
    """Every field the registry contract checks for other sources.

    The four pre-pioneer callables, the three fields the completeness
    ratchet requires on EVERY registered source (readiness probe,
    in-process dispatch, run summarizer), the work-unit kind, the
    snapshot state-key adapter Exa's live ``finalize_run_snapshot``
    path needs, and the two administrative flags.
    """

    entry = get_launcher("exa")
    assert isinstance(entry, LauncherEntry)

    for field in (
        "state_key_fn",
        "state_dir_fn",
        "orchestrator_argv_fn",
        "save_destination_blocker_fn",
        "readiness_probe_fn",
        "in_process_dispatch_fn",
        "summarize_run_fn",
        "snapshot_state_key_fn",
    ):
        value = getattr(entry, field)
        assert value is not None, f"exa.{field} must be populated"
        assert callable(value), f"exa.{field} must be callable"

    assert entry.progress_kind == EXA_QUERY_KIND
    assert entry.pipeline_state == "partial"
    assert entry.launchable is True
    assert entry.sunset is False

    # Deliberately unpopulated — see the registry entry's comment. A
    # future slice that fills one of these must also move it in
    # tests/test_launcher_registry_completeness.py's ratchet.
    assert entry.progress_projection_fn is None
    assert entry.form_strategy_fn is None
    assert entry.research_packet_builder_fn is None
    assert entry.brief_id_for_snapshot_fn is None
    assert entry.snapshot_research_packet_fn is None
    assert entry.reconstruct_report_analysis is False


def test_exa_orchestrator_argv_shape(tmp_path: Path) -> None:
    """argv matches ``exa/run.py:build_parser``'s contract.

    ``-u`` is load-bearing, not decoration: the launcher redirects a
    detached run's stdout to a log file and a buffered child shows the
    operator nothing for minutes. It also makes the prefix four
    elements — asserted here because ``_exa_in_process_dispatch``
    slices on exactly that number.
    """

    entry = get_launcher("exa")
    brief_path = _write_brief(tmp_path / "brief.json")
    state_dir = tmp_path / "state"

    argv = entry.orchestrator_argv_fn(
        str(brief_path), str(state_dir), resume=False
    )

    assert argv[0] == sys.executable
    assert argv[1] == "-u"
    assert argv[2] == "-m"
    assert argv[3] == "exa.run"
    assert argv[4:] == [
        "--brief",
        str(brief_path),
        "--state-dir",
        str(state_dir),
    ]
    assert "--resume" not in argv


def test_exa_orchestrator_argv_includes_resume_flag(tmp_path: Path) -> None:
    entry = get_launcher("exa")
    brief_path = _write_brief(tmp_path / "brief.json")
    state_dir = tmp_path / "state"

    argv = entry.orchestrator_argv_fn(
        str(brief_path), str(state_dir), resume=True
    )

    assert argv[-1] == "--resume"


def test_exa_orchestrator_argv_is_parseable_by_the_real_cli(
    tmp_path: Path,
) -> None:
    """The registry's argv and the orchestrator's argparse agree.

    Asserting the list shape alone would pass on an argv the CLI
    rejects. Parse it with the actual parser instead — this is the
    seam ``cloris.worker`` execvp's into.
    """

    from exa.run import build_parser

    entry = get_launcher("exa")
    brief_path = _write_brief(tmp_path / "brief.json")
    state_dir = tmp_path / "state"

    argv = entry.orchestrator_argv_fn(
        str(brief_path), str(state_dir), resume=True
    )
    args = build_parser().parse_args(argv[4:])

    assert args.brief == str(brief_path)
    assert args.state_dir == str(state_dir)
    assert args.resume is True


def test_exa_state_dir_lives_under_state_exa(tmp_path: Path) -> None:
    entry = get_launcher("exa")
    brief_path = _write_brief(tmp_path / "brief.json")

    state_dir = entry.state_dir_fn(str(brief_path))

    assert state_dir.parent.name == "exa"
    # The V2 loader derives the compat Brief's ``id`` from role_title
    # (shared/brief_loader.py:783), so that — not the JSON's ``id`` —
    # is what the state key slugifies.
    assert state_dir.name == entry.state_key_fn(str(brief_path))
    assert state_dir.name == "chief_of_staff_fixture"


def test_exa_blocker_none_for_clean_brief(tmp_path: Path) -> None:
    entry = get_launcher("exa")
    brief_path = _write_brief(tmp_path / "brief.json")

    assert entry.save_destination_blocker_fn(str(brief_path)) is None


def test_exa_blocker_none_for_unreadable_brief(tmp_path: Path) -> None:
    """An unreadable brief is the validation layer's 422, not this gate's.

    Same posture as ``_github_save_destination_blocker``: returning a
    blocker here would report a parse error as a save-destination
    problem and send the recruiter to the wrong field.
    """

    entry = get_launcher("exa")
    broken = tmp_path / "broken.json"
    broken.write_text("{ this is not json")

    assert entry.save_destination_blocker_fn(str(broken)) is None
    assert entry.save_destination_blocker_fn(str(tmp_path / "absent.json")) is None


def test_exa_blocker_names_first_blocking_lint_finding(tmp_path: Path) -> None:
    """A brief targeting a blacklisted firm blocks with the lint's own words.

    CLO-115 is a binding ruling: a current employee of a blacklisted
    firm is never a lead, so targeting the firm is an error, not a
    warning. The blocker must carry the lint's message (so the launch
    surface and the readiness surface say the same thing) plus a
    remediation naming the field to edit.
    """

    entry = get_launcher("exa")
    brief_path = _write_brief(
        tmp_path / "brief.json",
        exa_target_firms=[
            {"name": "Benchmark", "domain": "benchmark.com"},
            {"name": "OpenAI", "domain": "openai.com"},
        ],
    )

    blocker = entry.save_destination_blocker_fn(str(brief_path))

    assert isinstance(blocker, SaveDestinationBlocker)
    assert blocker.kind == "config"
    assert blocker.code == "employer_conflict"
    assert "OpenAI" in blocker.message
    assert "employer blacklist" in blocker.message
    assert "exa_target_firms" in blocker.remediation


def test_exa_blocker_ignores_non_blocking_findings(tmp_path: Path) -> None:
    """Info and warning findings are launch-legal.

    'IVP' trips the info-level ``namesake_risk`` rule (short one-word
    firm name, no exclusions). A launcher that refused on it would
    block half the real firm lists.
    """

    from exa.lane_compiler import lint_exa_brief
    from shared.brief_loader import load_brief

    entry = get_launcher("exa")
    brief_path = _write_brief(
        tmp_path / "brief.json",
        exa_target_firms=[{"name": "IVP", "domain": "ivp.com"}],
    )

    findings = lint_exa_brief(load_brief(str(brief_path)))
    assert any(f.code == "namesake_risk" for f in findings), (
        "fixture no longer trips the info-level rule this test is about"
    )
    assert entry.save_destination_blocker_fn(str(brief_path)) is None


@pytest.mark.skip(reason="Exercises retired cloris.worker module excluded from snapshot")
def test_exa_in_process_dispatch_hands_main_the_right_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The frozen-app path survives Exa's four-element argv prefix.

    ``_dispatch_orchestrator_in_process`` slices a fixed number of
    leading elements off the argv the registry produced. Exa's ``-u``
    makes that four, not three; with the default slice the module name
    would be left in the argument list and ``exa.run``'s argparse would
    reject it as an unrecognized positional. Asserting the shape alone
    would not catch that — run the dispatcher and read what ``main()``
    actually saw.
    """

    import sys as _sys

    from cloris import worker as worker_mod
    from exa.run import build_parser

    captured: dict[str, list[str]] = {}

    def fake_main() -> int:
        captured["argv"] = list(_sys.argv)
        return 7

    monkeypatch.setitem(
        _sys.modules,
        "exa.run",
        type("FakeExaRun", (), {"main": staticmethod(fake_main)}),
    )

    brief_path = _write_brief(tmp_path / "brief.json")
    state_dir = tmp_path / "state"
    argv = LAUNCHERS["exa"].orchestrator_argv_fn(
        str(brief_path), str(state_dir), resume=False
    )

    rc = worker_mod._dispatch_in_process("exa", argv)

    assert rc == 7, "dispatch fell through to the unknown-source branch"
    assert captured["argv"] == [
        "exa.run",
        "--brief",
        str(brief_path),
        "--state-dir",
        str(state_dir),
    ]

    # And the CLI that would really be on the other end accepts it.
    parsed = build_parser().parse_args(captured["argv"][1:])
    assert parsed.brief == str(brief_path)


def test_exa_readiness_probe_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry calls ``exa.health.probe_exa_readiness``, not a copy.

    Monkeypatching the module attribute proves the adapter resolves it
    per call (late import) rather than binding it at registry-import
    time — the property that lets the readiness surface be tested and
    that keeps ``cloris.launchers`` free of an ``exa`` import at module
    scope. No network, no EXA_API_KEY.
    """

    import exa.health as exa_health

    sentinel = object()
    calls: list[int] = []

    def fake_probe():
        calls.append(1)
        return sentinel

    monkeypatch.setattr(exa_health, "probe_exa_readiness", fake_probe)

    result = LAUNCHERS["exa"].readiness_probe_fn()

    assert result is sentinel
    assert calls == [1]


def test_exa_registry_import_does_not_import_the_exa_package() -> None:
    """Dependency direction: source modules register INTO the registry.

    ``cloris/launchers/__init__.py`` must carry no module-scope ``exa``
    import — the adapters late-import inside their bodies, the way
    github's do. An import at module scope would drag aiohttp / certifi
    into every process that touches the registry.
    """

    source = Path("cloris/launchers/__init__.py").read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import exa", "from exa")):
            assert line.startswith((" ", "\t")), (
                f"module-scope exa import in cloris/launchers: {line!r}"
            )
