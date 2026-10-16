"""Exa source module S5 — the CLI's refusals and the ``--check`` preflight.

Two contracts, and both are about what does NOT happen.

The preflight writes nothing: no store, no state directory, and — when
the brief cannot launch — not even the readiness probe, which is a real
billed Exa search (`exa/health.py:44-46`). A preflight that writes state
is a preflight you cannot run twice and get the same answer from, and
one that spends on a brief it has already rejected is a preflight that
costs money to be told no.

The launch refuses a fresh/resume mismatch before touching anything.
Fresh-vs-resume is a campaign decision and Sam's to make: a fresh run
over a live state dir re-runs a plan somebody already paid for, and a
``--resume`` typo against an empty dir would silently become a fresh
launch.

Exit 2 means REFUSED — nothing happened and the operator can fix the
command. Exit 1 means the run moved state and then failed. The tests
pin the difference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from exa.run import EXIT_ERROR, EXIT_OK, EXIT_REFUSED, main
from shared.health_types import ReadinessBlocker, ReadinessReport


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "exa_target_firms": [
            {"name": "Benchmark", "domain": "benchmark.com"},
            {"name": "Greylock", "domain": "greylock.com"},
        ],
        "exa_title_terms": ["Senior Associate", "Principal"],
    }
    raw.update(overrides)
    return raw


def _write_brief(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_brief_dict(**overrides)))
    return path


def _store_with_a_run(state_dir: Path) -> Path:
    """A state dir that really holds a run — a ``runs`` ROW, not a file.

    An empty placeholder file used to stand in for this. It cannot any
    more, and that is the point of the distinction: a database with no
    run rows is what a crash between ``initialize()`` and the first
    ``start_run`` leaves behind, and it means there is nothing to
    resume. These tests are about the OTHER case — a live campaign —
    so they have to build one.
    """

    from shared.runtime_state.store import RuntimeStateStore

    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    store.initialize()
    store.start_run(
        source="exa",
        brief_id="cos",
        output_dir=str(state_dir),
        mode="fresh",
    )
    return db_path


def _store_without_a_run(state_dir: Path) -> Path:
    """An initialized store that never started a run — the crash shape."""

    from shared.runtime_state.store import RuntimeStateStore

    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "runtime_state.sqlite3"
    RuntimeStateStore(db_path).initialize()
    return db_path


class _ProbeSpy:
    """A readiness probe that records whether it was called at all."""

    def __init__(self, ready: bool = True):
        self.calls = 0
        self.report = (
            ReadinessReport(ready=True, blockers=())
            if ready
            else ReadinessReport(
                ready=False,
                blockers=(
                    ReadinessBlocker(
                        kind="auth",
                        message="Exa rejected your API key.",
                        remediation="…",
                    ),
                ),
            )
        )

    def __call__(self, **_kwargs) -> ReadinessReport:
        self.calls += 1
        return self.report


def _probe(monkeypatch, ready: bool = True) -> _ProbeSpy:
    import exa.health

    spy = _ProbeSpy(ready=ready)
    monkeypatch.setattr(exa.health, "probe_exa_readiness", spy)
    return spy


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------


def test_check_mode_writes_no_state_and_exits_zero(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"

    exit_code = main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    assert exit_code == EXIT_OK
    assert not state_dir.exists(), "--check creates nothing, not even the state dir"

    out = capsys.readouterr().out
    assert "[exa] 2 queries across 2 firms × 1 lane" in out
    assert "[exa] Exa ready" in out
    assert f"[exa] state dir {state_dir}" in out
    assert "empty — this would be a fresh run" in out
    assert "run budget cap $" in out
    for line in out.splitlines():
        assert not line.lstrip().startswith("{")


def test_check_mode_plan_line_counts_archetypes_separately(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The archetype lane is not a factor in the firm × lane product.

    Twenty-six queries is 20 firms × 1 lane PLUS 6 archetypes, and a
    line that folded the archetype lane into the "× N lanes" factor
    would print arithmetic that does not work (20 × 2 ≠ 26) and read as
    a bug in the compiler.
    """

    _probe(monkeypatch)
    brief = _write_brief(
        tmp_path,
        exa_query_lanes=["people_search", "archetype"],
        exa_archetype_queries=[
            {"id": "a01-mbb-to-vc", "text": "Bay Area investors who consulted first."},
            {"id": "a02-investor-to-cos", "text": "Bay Area investors turned operator."},
        ],
    )
    state_dir = tmp_path / "state"

    assert main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"]) == (
        EXIT_OK
    )

    out = capsys.readouterr().out
    assert "[exa] 4 queries across 2 firms × 1 lane + 2 archetypes" in out
    # Still one line per event, still no raw JSON.
    for line in out.splitlines():
        assert not line.lstrip().startswith("{")


def test_check_mode_blocking_lint_skips_probe(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """CLO-115 blocks the launch — and the probe is not worth spending.

    The readiness check is a real Exa search. A brief that cannot launch
    has not earned one, so the preflight stops at the finding.
    """

    spy = _probe(monkeypatch)
    brief = _write_brief(tmp_path, employer_blacklist=["Benchmark"])
    state_dir = tmp_path / "state"

    exit_code = main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    assert exit_code == EXIT_REFUSED
    assert spy.calls == 0, "the billed probe must not fire behind a blocking finding"
    assert not state_dir.exists()

    out = capsys.readouterr().out
    assert "[exa] ERROR employer_conflict:" in out
    assert "Benchmark" in out
    assert "NO-GO — blocking lint finding(s); nothing else was checked" in out
    assert "Exa ready" not in out


def test_check_mode_reports_a_state_dir_that_holds_a_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)

    exit_code = main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    assert exit_code == EXIT_OK
    assert "holds a run — launch with --resume to continue it" in capsys.readouterr().out


def test_check_mode_distinguishes_a_database_from_a_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A store with no run rows is reported as fresh, not as a run.

    ``--check`` and the launch path must never disagree about the
    posture. Reporting "holds a run" here would send the operator to
    ``resume``, which the launch would then downgrade to fresh — a
    preflight that gives the wrong command is worse than one that says
    nothing.
    """

    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_without_a_run(state_dir)

    exit_code = main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert "database but no run — this would be a fresh run" in out
    assert "launch with --resume" not in out


def test_check_mode_exit_one_when_exa_is_not_ready(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _probe(monkeypatch, ready=False)
    brief = _write_brief(tmp_path)

    exit_code = main(
        ["--brief", str(brief), "--state-dir", str(tmp_path / "state"), "--check"]
    )

    assert exit_code == EXIT_ERROR
    assert "[exa] NOT READY auth: Exa rejected your API key." in capsys.readouterr().out


def test_check_mode_creates_no_directories(tmp_path: Path) -> None:
    """The import path itself must not touch the disk.

    Run in a SUBPROCESS, because the defect this pins was an import-time
    ``EXA_STATE_ROOT.mkdir()`` in ``exa/config.py`` — an in-process test
    would find the module already imported by the session and see
    nothing. ``CLORIS_USER_DATA_DIR`` redirects ``shared.config
    .OUTPUT_DIR`` (tests/conftest.py:34-67 uses the same seam), and
    ``EXA_API_KEY=""`` makes the readiness probe short-circuit on the
    missing key rather than reaching the network
    (`exa/health.py:75-87`).
    """

    brief = _write_brief(tmp_path)
    user_data = tmp_path / "userdata"
    env = {
        **os.environ,
        "CLORIS_USER_DATA_DIR": str(user_data),
        "EXA_API_KEY": "",
        "CLORIS_SKIP_STARTUP_VALIDATION": "1",
    }

    result = subprocess.run(
        [sys.executable, "-m", "exa.run", "--brief", str(brief), "--check"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == EXIT_ERROR, result.stdout + result.stderr
    assert "No Exa API key configured" in result.stdout
    # ``shared/output_paths.py`` scaffolds ``output/{state,runs,…}`` at
    # import (`:75-81`) — pre-existing repo behaviour, not Exa's. What
    # must NOT exist is the Exa half: no ``output/state/exa`` root (the
    # import-time mkdir this pins) and no per-brief state dir under it.
    exa_root = user_data / "output" / "state" / "exa"
    assert not exa_root.exists(), (
        "--check created output/state/exa; the import-time mkdir is back"
    )
    # And the preflight's own state-dir preview stayed a preview.
    assert not list((user_data / "output" / "state").glob("exa*"))


# ---------------------------------------------------------------------------
# Launch posture — fresh vs resume
# ---------------------------------------------------------------------------


def test_resume_without_a_store_is_refused(tmp_path: Path, capsys) -> None:
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    exit_code = main(
        ["--brief", str(brief), "--state-dir", str(state_dir), "--resume"]
    )

    assert exit_code == EXIT_REFUSED
    out = capsys.readouterr().out
    assert "NO-GO — resume requested but" in out
    assert "has no runtime_state.sqlite3" in out
    assert not (state_dir / "runtime_state.sqlite3").exists()


def test_fresh_run_over_an_existing_store_is_refused(tmp_path: Path, capsys) -> None:
    """A fresh launch over a live campaign is a re-spend, not a restart."""

    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    db_path = _store_with_a_run(state_dir)
    before = db_path.read_bytes()

    exit_code = main(["--brief", str(brief), "--state-dir", str(state_dir)])

    assert exit_code == EXIT_REFUSED
    out = capsys.readouterr().out
    assert "already holds a run; use --resume" in out
    # Refused before anything ran: the store is byte-identical (the
    # posture read opens it mode=ro) and no projection landed beside it.
    assert db_path.read_bytes() == before
    assert not (state_dir / "progress.json").exists()


def test_resume_on_empty_db_runs_fresh_with_one_line(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A database with no run rows is not something to resume.

    This is the crash-after-init shape: ``initialize()`` ran, the first
    ``start_run`` never did. There is no query plan, no candidate, no
    verdict. Refusing ``--resume`` would strand the campaign on an
    operator who has to know to delete a file; honoring it would run
    resume semantics — the pending-judgment pre-pass, the cloned work
    units — over a plan that was never written. It runs FRESH, and says
    so in one line, because a mode the operator did not ask for must
    not be silent.
    """

    captured: dict = {}

    class _FakePipeline:
        def __init__(self, **kwargs):
            captured["ctor"] = kwargs

        async def run(self, *, resume, max_queries=None):
            captured["resume"] = resume

            class _Stats:
                run_id = 1
                judged = saved = review = rejected = 0

            return _Stats()

    import exa.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "ExaPipeline", _FakePipeline)

    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_without_a_run(state_dir)

    exit_code = main(
        ["--brief", str(brief), "--state-dir", str(state_dir), "--resume"]
    )

    assert exit_code == EXIT_OK
    # The pipeline ran, and it ran FRESH — not "resume" over an empty plan.
    assert captured["resume"] is False

    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if "database but no run" in line
    ]
    assert lines == [
        "    [exa] state dir has a database but no run — starting fresh"
    ]


def test_resume_on_a_real_run_stays_a_resume(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The fallback must not swallow a legitimate resume.

    Downgrading a real resume to fresh would re-run a paid query plan —
    the exact failure the posture check exists to prevent — so the
    fallback is pinned from both sides.
    """

    captured: dict = {}

    class _FakePipeline:
        def __init__(self, **kwargs):
            pass

        async def run(self, *, resume, max_queries=None):
            captured["resume"] = resume

            class _Stats:
                run_id = 1
                judged = saved = review = rejected = 0

            return _Stats()

    import exa.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "ExaPipeline", _FakePipeline)

    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)

    exit_code = main(
        ["--brief", str(brief), "--state-dir", str(state_dir), "--resume"]
    )

    assert exit_code == EXIT_OK
    assert captured["resume"] is True
    assert "database but no run" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Bad input
# ---------------------------------------------------------------------------


def test_unreadable_brief_is_refused_with_one_line(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "brief.json"
    bad.write_text("{ this is not json")

    exit_code = main(["--brief", str(bad), "--state-dir", str(tmp_path / "state")])

    assert exit_code == EXIT_REFUSED
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("    [exa] NO-GO — cannot read the brief:")


def test_missing_brief_file_is_refused_in_check_mode(tmp_path: Path, capsys) -> None:
    exit_code = main(
        ["--brief", str(tmp_path / "nope.json"), "--state-dir", str(tmp_path), "--check"]
    )

    assert exit_code == EXIT_REFUSED
    assert "NO-GO — cannot read the brief:" in capsys.readouterr().out


def test_blocking_lint_on_a_real_launch_is_refused(
    tmp_path: Path, capsys
) -> None:
    """``ExaBriefBlockedError`` is a refusal (2), not a run failure (1)."""

    brief = _write_brief(tmp_path, employer_blacklist=["Benchmark"])
    state_dir = tmp_path / "state"

    exit_code = main(["--brief", str(brief), "--state-dir", str(state_dir)])

    assert exit_code == EXIT_REFUSED
    out = capsys.readouterr().out
    assert "[exa] BLOCKED employer_conflict:" in out
    assert not (state_dir / "runtime_state.sqlite3").exists()


# ---------------------------------------------------------------------------
# --check states the spend a resume starts from (2026-10-10)
# ---------------------------------------------------------------------------


def _receipt(actual, estimate=0.297):
    return json.dumps(
        {
            "schema_version": "exa-usage-v1",
            "endpoint": "/search",
            "estimate_usd": estimate,
            "actual_usd": actual,
            "usage_status": "measured" if actual is not None else "unavailable",
        }
    )


def test_check_mode_reports_measured_spend_and_the_resumed_meter_start(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The ledger is the spend; a resumed session's meter starts at zero.

    The overnight SPL run's console said $9.94 while exa-usage.jsonl held
    $0.744. --check reads the receipts, never the dead session's meter.
    """

    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)
    ledger = state_dir / "exa-usage.jsonl"
    ledger.write_text("\n".join([_receipt(0.647), _receipt(0.097)]) + "\n")
    before = {p.name: p.stat().st_mtime_ns for p in state_dir.iterdir()}

    exit_code = main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert (
        "[exa] Exa spend to date $0.7440 across 2 receipts (exa-usage.jsonl); "
        "a resumed session's meter starts at $0.0000 of the $5.00 cap"
    ) in out
    after = {p.name: p.stat().st_mtime_ns for p in state_dir.iterdir()}
    assert after == before, "--check wrote into the state dir"


def test_check_mode_counts_an_unmeasured_receipt_at_its_estimate(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)
    (state_dir / "exa-usage.jsonl").write_text(
        "\n".join([_receipt(0.007), _receipt(None, estimate=0.027)]) + "\n"
    )

    main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    assert "Exa spend to date $0.0340 across 2 receipts" in capsys.readouterr().out


def test_check_mode_without_a_ledger_says_nothing_was_spent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"

    main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    out = capsys.readouterr().out
    assert "[exa] Exa spend to date $0.0000 (no exa-usage.jsonl yet)" in out
    assert not state_dir.exists()


def test_posture_check_on_a_closed_store_creates_no_sqlite_siblings(
    tmp_path: Path,
) -> None:
    from exa.run import _state_dir_holds_a_run

    state_dir = tmp_path / "state"
    db_path = _store_with_a_run(state_dir)
    for suffix in ("-wal", "-shm"):
        sibling = db_path.with_name(db_path.name + suffix)
        if sibling.exists():
            sibling.unlink()
    before = sorted(p.name for p in state_dir.iterdir())

    assert _state_dir_holds_a_run(state_dir) is True
    assert sorted(p.name for p in state_dir.iterdir()) == before


def test_posture_check_sees_a_run_row_still_in_the_wal(tmp_path: Path) -> None:
    """A live writer's first run row may exist only in the -wal file."""

    import sqlite3

    from exa.run import _state_dir_holds_a_run

    state_dir = tmp_path / "state"
    db_path = _store_without_a_run(state_dir)
    writer = sqlite3.connect(db_path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO runs (source, brief_id, output_dir, mode, status, started_at) "
            "VALUES ('exa', 'cos', ?, 'fresh', 'running', '2026-10-10T00:00:00Z')",
            (str(state_dir),),
        )
        writer.commit()
        assert db_path.with_name(db_path.name + "-wal").exists()

        assert _state_dir_holds_a_run(state_dir) is True
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# The location scope on the operator surfaces
# ---------------------------------------------------------------------------


def test_check_mode_names_the_location_scope(tmp_path: Path, monkeypatch, capsys) -> None:
    _probe(monkeypatch)
    brief = _write_brief(tmp_path, exa_location_scope={"countries": ["United States"]})

    main(["--brief", str(brief), "--state-dir", str(tmp_path / "state"), "--check"])

    assert (
        "[exa] location scope on: the United States; a card location outside it "
        "is excluded before judgment, an unreadable one goes to the judge"
    ) in capsys.readouterr().out


def test_check_mode_without_a_location_scope_says_nothing_about_it(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _probe(monkeypatch)
    brief = _write_brief(tmp_path)

    main(["--brief", str(brief), "--state-dir", str(tmp_path / "state"), "--check"])

    assert "location scope" not in capsys.readouterr().out


def test_run_summary_counts_location_exclusions(tmp_path: Path, monkeypatch, capsys) -> None:
    class _FakePipeline:
        def __init__(self, **kwargs):
            from exa.location_gate import LocationScope

            self.location_scope = LocationScope.from_brief(
                {"exa_location_scope": {"countries": ["United States"]}}
            )

        async def run(self, *, resume, max_queries=None):
            class _Stats:
                run_id = 7
                judged = saved = review = rejected = 0
                location_gated = 3

            return _Stats()

    import exa.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "ExaPipeline", _FakePipeline)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)

    assert main(["--brief", str(brief), "--state-dir", str(state_dir), "--resume"]) == EXIT_OK

    assert (
        "3 excluded before judgment (outside the United States)"
        in capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# --check and the run summary state the re-judge a resume will do (2026-10-10)
# ---------------------------------------------------------------------------


def _stored_verdict(
    db_path: Path,
    *,
    handle: str,
    decision: str,
    fingerprint: str,
    brief_id: str = "cos",
    path: str = "DIRECT:1. Operating leverage",
) -> None:
    """One judged person, written the way the pipeline's terminal write leaves it."""

    import sqlite3

    from shared.runtime_state.store import RuntimeStateStore

    RuntimeStateStore(db_path).ensure_candidate(
        source="exa",
        brief_id=brief_id,
        identity_key=f"linkedin.com/in/{handle}",
        display_name=handle,
        profile_url=f"https://www.linkedin.com/in/{handle}",
    )
    payload = {
        "full_decision": {
            "stage": "full",
            "decision": decision,
            "path": path,
            "confidence": 0.7,
            "rationale": "scripted",
            "candidate_name": handle,
            "profile_url": f"https://www.linkedin.com/in/{handle}",
        },
        "candidate_record": {"name": handle, "linkedin_url": f"https://www.linkedin.com/in/{handle}"},
        "judgment_fingerprint": fingerprint,
    }
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE candidates SET current_lifecycle_state = 'full_terminal', "
            "terminal_decision = ?, terminal_payload_json = ?, "
            "brief_content_hash = 'sha256:old' WHERE identity_key = ?",
            (decision, json.dumps(payload), f"linkedin.com/in/{handle}"),
        )
        conn.commit()
    finally:
        conn.close()


def test_check_mode_counts_slate_verdicts_a_resume_will_rejudge(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from exa.judge import judgment_fingerprint
    from shared.brief_loader import load_brief
    from shared.output_paths import derive_brief_id

    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    brief_id = str(derive_brief_id(brief_path=brief, brief=load_brief(str(brief))))
    current = judgment_fingerprint(load_brief(str(brief)))
    state_dir = tmp_path / "state"
    db_path = _store_with_a_run(state_dir)
    from exa.location_gate import LOCATION_GATE_PATH

    for handle, decision, fingerprint, path in (
        ("old-save", "SAVE", "sha256:old", ""),
        ("old-inferential", "INFERENTIAL_SAVE", "sha256:old", ""),
        ("old-review", "REVIEW_FLAGGED", "sha256:old", ""),
        ("old-reject", "REJECT", "sha256:old", ""),
        # A gate's exclusion is not a verdict about criteria: not counted.
        ("old-gated", "REJECT", "", LOCATION_GATE_PATH),
        ("current-save", "SAVE", current, ""),
    ):
        _stored_verdict(
            db_path,
            handle=handle,
            decision=decision,
            fingerprint=fingerprint,
            brief_id=brief_id,
            **({"path": path} if path else {}),
        )
    for suffix in ("-wal", "-shm"):
        sibling = db_path.with_name(db_path.name + suffix)
        if sibling.exists():
            sibling.unlink()
    before = {p.name: p.stat().st_mtime_ns for p in state_dir.iterdir()}

    exit_code = main(
        ["--brief", str(brief), "--state-dir", str(state_dir), "--check", "--rejudge-stale"]
    )

    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert (
        "[exa] 3 slate verdicts were judged under earlier judging criteria; a resume "
        "re-judges them first, two samples each (model spend only, no Exa call); 1 "
        "earlier reject keeps its verdict (up to 6 judge calls)"
    ) in out
    after = {p.name: p.stat().st_mtime_ns for p in state_dir.iterdir()}
    assert after == before, "--check wrote into the state dir"


def _stale_campaign(tmp_path: Path) -> tuple[Path, Path, str]:
    from exa.judge import judgment_fingerprint
    from shared.brief_loader import load_brief
    from shared.output_paths import derive_brief_id

    brief = _write_brief(tmp_path)
    loaded = load_brief(str(brief))
    brief_id = str(derive_brief_id(brief_path=brief, brief=loaded))
    state_dir = tmp_path / "state"
    db_path = _store_with_a_run(state_dir)
    for handle, decision in (("old-save", "SAVE"), ("old-reject", "REJECT")):
        _stored_verdict(
            db_path, handle=handle, decision=decision, fingerprint="sha256:old",
            brief_id=brief_id,
        )
    return brief, state_dir, judgment_fingerprint(loaded)


def test_check_mode_states_the_fingerprint_and_which_opt_ins_are_on(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Review round 3: --check tells the truth about what a resume will re-judge."""

    _probe(monkeypatch)
    brief, state_dir, fingerprint = _stale_campaign(tmp_path)
    base = ["--brief", str(brief), "--state-dir", str(state_dir), "--check"]

    main([*base, "--rejudge-stale-rejects"])
    out = capsys.readouterr().out
    assert (
        f"[exa] judgment fingerprint {fingerprint} (the current brief); re-judge on "
        "resume: slate verdicts on (--rejudge-stale), earlier rejects on "
        "(--rejudge-stale-rejects)"
    ) in out
    assert (
        "[exa] 1 slate verdict was judged under earlier judging criteria; a resume "
        "re-judges it first, two samples each (model spend only, no Exa call); 1 "
        "earlier reject is re-judged too, one sample (up to 3 judge calls)"
    ) in out

    main(base)
    out = capsys.readouterr().out
    assert (
        f"[exa] judgment fingerprint {fingerprint} (the current brief); re-judge on "
        "resume: slate verdicts off, earlier rejects off"
    ) in out
    assert (
        "[exa] 1 slate verdict was judged under earlier judging criteria; it keeps its "
        "verdict on resume (re-judge is off; --rejudge-stale turns it on); 1 earlier "
        "reject keeps its verdict"
    ) in out


def test_check_mode_is_silent_about_rejudging_when_nothing_is_stale(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _probe(monkeypatch)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)

    main(["--brief", str(brief), "--state-dir", str(state_dir), "--check"])

    assert "re-judges" not in capsys.readouterr().out


def test_run_summary_counts_rejudged_verdicts(tmp_path: Path, monkeypatch, capsys) -> None:
    class _FakePipeline:
        def __init__(self, **kwargs):
            self.location_scope = None

        async def run(self, *, resume, max_queries=None):
            class _Stats:
                run_id = 8
                judged = 3
                saved = review = rejected = 0
                rejudged = 3
                rejudge_left_short_list = 1

            return _Stats()

    import exa.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "ExaPipeline", _FakePipeline)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)

    assert main(["--brief", str(brief), "--state-dir", str(state_dir), "--resume"]) == EXIT_OK

    assert (
        ", 3 re-judged under the current criteria (1 left the short list)"
        in capsys.readouterr().out
    )


def test_run_summary_counts_location_exclusions_apart_from_leaving_the_short_list(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Review round 3: "(M left the short list)" counted people the location
    gate removed on re-judge, which read as the new criteria moving them."""

    class _FakePipeline:
        def __init__(self, **kwargs):
            from exa.location_gate import LocationScope

            self.location_scope = LocationScope.from_brief(
                {"exa_location_scope": {"countries": ["United States"]}}
            )

        async def run(self, *, resume, max_queries=None):
            class _Stats:
                run_id = 8
                judged = 190
                saved = review = rejected = 0
                rejudged = 653
                rejudge_left_short_list = 12
                rejudge_location_excluded = 5
                rejudge_blacklist_excluded = 0
                rejudge_split = 3
                rejudge_rejects = 463
                rejudge_promoted = 9
                rejudge_kept = 2

            return _Stats()

    import exa.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "ExaPipeline", _FakePipeline)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)

    assert main(["--brief", str(brief), "--state-dir", str(state_dir), "--resume"]) == EXIT_OK

    assert (
        ", 653 re-judged under the current criteria (12 left the short list, 3 split "
        "to worth a look, 9 of 463 earlier rejects promoted), 5 of them excluded on "
        "re-judge (outside the United States), 2 kept their stored verdict after a "
        "failed re-judge sample (tried again next resume)"
    ) in capsys.readouterr().out


def test_rejudge_opt_ins_reach_the_pipeline(tmp_path: Path, monkeypatch) -> None:
    """The re-judge pass runs only on the flags; --rejudge-stale-rejects implies the slate pass."""

    built: list[dict] = []

    class _FakePipeline:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.location_scope = None

        async def run(self, *, resume, max_queries=None):
            class _Stats:
                run_id = 8
                judged = saved = review = rejected = 0

            return _Stats()

    import exa.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "ExaPipeline", _FakePipeline)
    brief = _write_brief(tmp_path)
    state_dir = tmp_path / "state"
    _store_with_a_run(state_dir)
    base = ["--brief", str(brief), "--state-dir", str(state_dir), "--resume"]

    assert main(base) == EXIT_OK
    assert main([*base, "--rejudge-stale"]) == EXIT_OK
    assert main([*base, "--rejudge-stale-rejects"]) == EXIT_OK

    assert [
        (kwargs.get("rejudge_stale"), kwargs.get("rejudge_stale_rejects"))
        for kwargs in built
    ] == [(False, False), (True, False), (True, True)]
