"""Tests for ``tools/step_b_hold_report.py``.

Hard rules pinned by these tests (mirror slice 11's import-narrowness
and write-discipline guarantees):

- No live calls. No imports from ``linkedin/``, ``github/``,
  ``market_intelligence/``, ``shared/runtime_state``,
  ``shared/external_evidence``, ``shared/judger``, or
  ``shared/schemas`` (verified by string-search on the new module's
  source -- mirrors ``tests/test_discover_facial_gate_inputs.py``).
- ``_count_borderline_alias_rationales``, ``_count_flag_off_canary``,
  ``_count_facial_decisions``, ``_classify_run``, ``discover_runs``,
  ``build_aggregate_summary``, ``format_per_run_listing``, and
  ``format_aggregate_summary`` are pure: same input -> same output, no
  clocks, no randomness.
- The default invocation writes nothing to disk; only ``--report-out``
  writes, and it refuses paths under any protected ``output/`` subtree.
- Empty discovery exits 0 with the ``"No completed LinkedIn runs found"``
  message (the report is a query, not a hard requirement).
- The three verdicts (``PASS``, ``BLOCKED``, ``NOT_APPLICABLE``) and the
  blocking-gate names are pinned by the classification tests.
- The borderline canary is the unicode ``[BORDERLINE\u2192YES alias]``;
  the ASCII ``->`` form is explicitly NOT a match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import step_b_hold_report as hold  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_run_tree(
    *,
    runs_root: Path,
    source: str,
    brief_id: str,
    run_label: str,
    files: dict[str, bytes] | None = None,
) -> Path:
    """Build ``runs_root/<source>/<brief-id>/<run-label>/`` with ``files``.

    Mirrors slice 11's helper exactly. ``files`` is
    ``{filename: contents_bytes}``. Missing keys are not created. The
    four-part path satisfies ``shared.output_paths.is_run_dir`` when
    ``runs_root`` ends in ``output/runs``.
    """

    run_dir = runs_root / source / brief_id / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (files or {}).items():
        (run_dir / name).write_bytes(payload)
    return run_dir


def _runs_root(tmp_path: Path) -> Path:
    """Return a path that ``is_run_dir`` accepts as ``output/runs/...``.

    ``shared.output_paths._parts_after_output`` walks parents looking
    for a component literally named ``"output"`` and treats anything
    four-deep after that as a run-dir. We anchor on
    ``tmp_path/output/runs``.
    """

    root = tmp_path / "output" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ns(**overrides) -> argparse.Namespace:
    """Build a defaults-aligned argparse namespace for ``run()``."""

    base = dict(
        source="linkedin",
        brief_id=None,
        limit=20,
        report_out=None,
        include_legacy=False,
        include_unknown_brief=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _facial_row(
    *,
    decision: str = "FACIAL_NO",
    rationale: str = "stub rationale",
    candidate_name: str = "Test Candidate",
) -> bytes:
    """Build a single ``facial_judgments.jsonl`` row matching the on-disk shape.

    Mirrors the fields actually written to disk (verified against
    ``output/runs/linkedin/2009570906/2026-04-24...__run-6/
    facial_judgments.jsonl``).
    """

    return (
        json.dumps(
            {
                "candidate_name": candidate_name,
                "confidence": 1.0,
                "decision": decision,
                "novelty_value": "",
                "path": "none",
                "post_save_modifier": "NONE",
                "profile_url": "https://example/test",
                "rationale": rationale,
                "stage": "facial",
                "value_rationale": "",
            }
        ).encode("utf-8")
        + b"\n"
    )


def _full_pass_files(
    *,
    borderline_count: int = 1,
    yes_extra: int = 2,
    no_extra: int = 1,
    skip_extra: int = 1,
    flag_off_canary_count: int = 0,
) -> dict[str, bytes]:
    """Build a file-set whose run-dir would land at ``PASS``.

    All four required artifacts present and non-empty;
    ``facial_judgments.jsonl`` carries ``borderline_count`` aliased
    rows (which are persisted as ``decision=FACIAL_YES`` per the
    orchestrator) plus filler YES/NO/SKIP rows so totals are non-trivial.
    """

    rows = b""
    for i in range(borderline_count):
        rows += _facial_row(
            decision="FACIAL_YES",
            rationale=(
                f"{hold._BORDERLINE_ALIAS_PREFIX} aliased borderline #{i}"
            ),
        )
    for i in range(yes_extra):
        rows += _facial_row(
            decision="FACIAL_YES",
            rationale=f"clean YES rationale #{i}",
        )
    for i in range(no_extra):
        rows += _facial_row(
            decision="FACIAL_NO",
            rationale=f"clean NO rationale #{i}",
        )
    for i in range(skip_extra):
        rows += _facial_row(
            decision="FACIAL_SKIP",
            rationale=f"skip rationale #{i}",
        )
    for i in range(flag_off_canary_count):
        rows += _facial_row(
            decision="PARSE_FAILURE",
            rationale=(
                "Facial parser emitted FACIAL_BORDERLINE while flag off. "
                f"reason={hold._FLAG_OFF_CANARY_SUBSTRING}"
            ),
        )
    return {
        "snippets.jsonl": b'{"k": 1}\n',
        "profile_summaries.jsonl": b'{"k": 2}\n',
        "final_judgments.jsonl": b'{"k": 3}\n',
        "facial_judgments.jsonl": rows,
    }


# ---------------------------------------------------------------------------
# 1-4. _count_borderline_alias_rationales
# ---------------------------------------------------------------------------


def test_count_borderline_alias_rationales_happy_path(tmp_path):
    facial = tmp_path / "facial_judgments.jsonl"
    facial.write_bytes(
        _facial_row(
            decision="FACIAL_YES",
            rationale=f"{hold._BORDERLINE_ALIAS_PREFIX} thing one",
        )
        + _facial_row(decision="FACIAL_NO", rationale="not borderline")
        + _facial_row(
            decision="FACIAL_YES",
            rationale=f"{hold._BORDERLINE_ALIAS_PREFIX} thing two",
        )
    )
    assert hold._count_borderline_alias_rationales(facial) == 2


def test_count_borderline_alias_rationales_no_matches(tmp_path):
    facial = tmp_path / "facial_judgments.jsonl"
    facial.write_bytes(
        _facial_row(decision="FACIAL_YES", rationale="ordinary YES")
        + _facial_row(decision="FACIAL_NO", rationale="ordinary NO")
        + _facial_row(decision="FACIAL_SKIP", rationale="ordinary SKIP")
    )
    assert hold._count_borderline_alias_rationales(facial) == 0


def test_count_borderline_alias_rationales_tolerates_malformed(tmp_path):
    facial = tmp_path / "facial_judgments.jsonl"
    valid_one = _facial_row(
        decision="FACIAL_YES",
        rationale=f"{hold._BORDERLINE_ALIAS_PREFIX} valid first",
    )
    valid_two = _facial_row(
        decision="FACIAL_YES",
        rationale=f"{hold._BORDERLINE_ALIAS_PREFIX} valid second",
    )
    bogus = b"this is not json at all\n"
    facial.write_bytes(valid_one + bogus + valid_two)
    assert hold._count_borderline_alias_rationales(facial) == 2


def test_count_borderline_alias_rationales_unicode_arrow_only(tmp_path):
    """The canary is the unicode arrow; ASCII ``->`` must NOT be counted."""

    facial = tmp_path / "facial_judgments.jsonl"
    ascii_form = "[BORDERLINE->YES alias] decoy"
    unicode_form = f"{hold._BORDERLINE_ALIAS_PREFIX} real"
    facial.write_bytes(
        _facial_row(decision="FACIAL_YES", rationale=ascii_form)
        + _facial_row(decision="FACIAL_YES", rationale=unicode_form)
    )
    assert hold._count_borderline_alias_rationales(facial) == 1


# ---------------------------------------------------------------------------
# 5. _count_flag_off_canary
# ---------------------------------------------------------------------------


def test_count_flag_off_canary_picks_up_substring(tmp_path):
    facial = tmp_path / "facial_judgments.jsonl"
    facial.write_bytes(
        _facial_row(decision="FACIAL_YES", rationale="not the canary")
        + _facial_row(
            decision="PARSE_FAILURE",
            rationale=(
                "Facial parser emitted FACIAL_BORDERLINE while flag off. "
                f"reason={hold._FLAG_OFF_CANARY_SUBSTRING}"
            ),
        )
        + _facial_row(decision="FACIAL_NO", rationale="another clean row")
    )
    assert hold._count_flag_off_canary(facial) == 1


# ---------------------------------------------------------------------------
# 6. _count_facial_decisions
# ---------------------------------------------------------------------------


def test_count_facial_decisions_excludes_skip_from_open_rate_denominator(
    tmp_path,
):
    facial = tmp_path / "facial_judgments.jsonl"
    rows = (
        _facial_row(decision="FACIAL_YES")
        + _facial_row(decision="FACIAL_YES")
        + _facial_row(decision="FACIAL_YES")
        + _facial_row(decision="FACIAL_NO")
        + _facial_row(decision="FACIAL_SKIP")
    )
    facial.write_bytes(rows)
    total, yes, non_skip = hold._count_facial_decisions(facial)
    assert total == 5
    assert yes == 3
    assert non_skip == 4


# ---------------------------------------------------------------------------
# 7-13. _classify_run
# ---------------------------------------------------------------------------


def test_classify_run_pass_happy_path(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=_full_pass_files(borderline_count=2, yes_extra=3, no_extra=2),
    )
    report = hold._classify_run(run_dir)
    assert report.verdict == hold.VERDICT_PASS
    assert report.flag_was_on is True
    assert report.flag_off_canary_count == 0
    assert report.facial_borderline_count == 2
    assert report.total_facial_decisions == 8  # 2 alias + 3 yes + 2 no + 1 skip
    assert report.facial_yes_count == 5
    assert report.facial_non_skip_total == 7
    assert report.snippets_present is True
    assert report.profile_summaries_present is True
    assert report.final_judgments_present_and_nonempty is True
    assert report.facial_judgments_present is True
    assert report.eligible_for_step_b_hold is True
    assert report.blocking_gate == hold.GATE_NONE
    assert report.brief_id == "brief-a"
    assert report.source == "linkedin"
    assert report.facial_borderline_rate == pytest.approx(2 / 8)
    assert report.facial_open_rate == pytest.approx(5 / 7)


def test_classify_run_zero_byte_finals(tmp_path):
    """Mirror the pre-slice-18 reality of the existing 6 finalized runs."""

    runs_root = _runs_root(tmp_path)
    files = _full_pass_files(borderline_count=2)
    files["final_judgments.jsonl"] = b""  # zero bytes
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=files,
    )
    report = hold._classify_run(run_dir)
    assert report.verdict == hold.VERDICT_BLOCKED
    assert report.blocking_gate == hold.GATE_FINAL_JUDGMENTS_ZERO_BYTES
    assert report.flag_was_on is True
    assert report.eligible_for_step_b_hold is False
    assert report.final_judgments_present_and_nonempty is False


def test_classify_run_flag_never_on_returns_not_applicable(tmp_path):
    """No borderline canary AND no flag-off canary -> NOT_APPLICABLE."""

    runs_root = _runs_root(tmp_path)
    files = {
        "snippets.jsonl": b'{"k": 1}\n',
        "profile_summaries.jsonl": b'{"k": 2}\n',
        "final_judgments.jsonl": b'{"k": 3}\n',
        "facial_judgments.jsonl": (
            _facial_row(decision="FACIAL_YES", rationale="clean YES")
            + _facial_row(decision="FACIAL_NO", rationale="clean NO")
        ),
    }
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=files,
    )
    report = hold._classify_run(run_dir)
    assert report.verdict == hold.VERDICT_NOT_APPLICABLE
    assert report.flag_was_on is False
    assert report.blocking_gate == hold.GATE_FLAG_NEVER_ON
    assert report.eligible_for_step_b_hold is False


def test_classify_run_flag_off_canary_present(tmp_path):
    runs_root = _runs_root(tmp_path)
    files = _full_pass_files(borderline_count=0, flag_off_canary_count=1)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=files,
    )
    report = hold._classify_run(run_dir)
    assert report.verdict == hold.VERDICT_BLOCKED
    assert report.blocking_gate == hold.GATE_FLAG_OFF_CANARY_PRESENT
    assert report.flag_was_on is True
    assert report.flag_off_canary_count == 1
    assert report.eligible_for_step_b_hold is False


def test_classify_run_missing_facial_artifact(tmp_path):
    """No facial_judgments.jsonl at all -> NOT_APPLICABLE (flag never on).

    With no facial artifact we cannot observe either canary, so
    ``flag_was_on`` is False and the verdict is ``NOT_APPLICABLE``. This
    is distinct from the BLOCKED ``missing_facial_artifact`` gate which
    only fires when other signals already proved the flag was on.
    """

    runs_root = _runs_root(tmp_path)
    files = {
        "snippets.jsonl": b'{"k": 1}\n',
        "profile_summaries.jsonl": b'{"k": 2}\n',
        "final_judgments.jsonl": b'{"k": 3}\n',
    }
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=files,
    )
    report = hold._classify_run(run_dir)
    assert report.facial_judgments_present is False
    assert report.flag_was_on is False
    assert report.verdict == hold.VERDICT_NOT_APPLICABLE
    assert report.blocking_gate == hold.GATE_FLAG_NEVER_ON


def test_classify_run_missing_snippets(tmp_path):
    runs_root = _runs_root(tmp_path)
    files = _full_pass_files(borderline_count=2)
    files.pop("snippets.jsonl")
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=files,
    )
    report = hold._classify_run(run_dir)
    assert report.verdict == hold.VERDICT_BLOCKED
    assert report.blocking_gate == hold.GATE_MISSING_SNIPPETS
    assert report.snippets_present is False


def test_classify_run_missing_profile_summaries(tmp_path):
    runs_root = _runs_root(tmp_path)
    files = _full_pass_files(borderline_count=2)
    files.pop("profile_summaries.jsonl")
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=files,
    )
    report = hold._classify_run(run_dir)
    assert report.verdict == hold.VERDICT_BLOCKED
    assert report.blocking_gate == hold.GATE_MISSING_PROFILE_SUMMARIES
    assert report.profile_summaries_present is False


# ---------------------------------------------------------------------------
# 14. discover_runs filters
# ---------------------------------------------------------------------------


def test_discover_runs_skips_legacy_and_unknown_by_default(tmp_path):
    runs_root = _runs_root(tmp_path)
    real = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"facial_judgments.jsonl": b""},
    )
    legacy = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="imported-2026-04-08T22-44-23-171843+00-00__legacy-1",
        files={"facial_judgments.jsonl": b""},
    )
    unknown = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="unknown",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"facial_judgments.jsonl": b""},
    )

    default = hold.discover_runs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=False,
        include_unknown_brief=False,
        brief_id_filter=None,
    )
    assert real in default
    assert legacy not in default
    assert unknown not in default

    with_legacy = hold.discover_runs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=True,
        include_unknown_brief=False,
        brief_id_filter=None,
    )
    assert legacy in with_legacy

    with_unknown = hold.discover_runs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=False,
        include_unknown_brief=True,
        brief_id_filter=None,
    )
    assert unknown in with_unknown


# ---------------------------------------------------------------------------
# 15. build_aggregate_summary
# ---------------------------------------------------------------------------


def _stub_report(
    *, brief_id: str, run_label: str, verdict: str
) -> hold.RunHoldReport:
    return hold.RunHoldReport(
        source="linkedin",
        brief_id=brief_id,
        run_dir=Path("/dev/null") / brief_id / run_label,
        run_label=run_label,
        flag_was_on=verdict != hold.VERDICT_NOT_APPLICABLE,
        flag_off_canary_count=0,
        total_facial_decisions=10,
        facial_borderline_count=1 if verdict == hold.VERDICT_PASS else 0,
        facial_borderline_rate=0.1 if verdict == hold.VERDICT_PASS else 0.0,
        facial_yes_count=4,
        facial_non_skip_total=8,
        facial_open_rate=0.5,
        snippets_present=True,
        profile_summaries_present=True,
        final_judgments_present_and_nonempty=verdict == hold.VERDICT_PASS,
        facial_judgments_present=True,
        bias_monitor_checkpoint_present=True,
        eligible_for_step_b_hold=verdict == hold.VERDICT_PASS,
        verdict=verdict,
        blocking_gate=(
            hold.GATE_NONE
            if verdict == hold.VERDICT_PASS
            else hold.GATE_FINAL_JUDGMENTS_ZERO_BYTES
            if verdict == hold.VERDICT_BLOCKED
            else hold.GATE_FLAG_NEVER_ON
        ),
    )


def test_build_aggregate_summary_counts_and_by_brief():
    reports = [
        _stub_report(brief_id="brief-a", run_label="run-1", verdict=hold.VERDICT_PASS),
        _stub_report(brief_id="brief-a", run_label="run-2", verdict=hold.VERDICT_BLOCKED),
        _stub_report(brief_id="brief-b", run_label="run-1", verdict=hold.VERDICT_BLOCKED),
        _stub_report(brief_id="brief-b", run_label="run-2", verdict=hold.VERDICT_NOT_APPLICABLE),
    ]
    summary = hold.build_aggregate_summary(reports)
    assert summary["total_runs_discovered"] == 4
    assert summary[hold.VERDICT_PASS] == 1
    assert summary[hold.VERDICT_BLOCKED] == 2
    assert summary[hold.VERDICT_NOT_APPLICABLE] == 1
    assert set(summary["by_brief_id"].keys()) == {"brief-a", "brief-b"}
    assert summary["by_brief_id"]["brief-a"][hold.VERDICT_PASS] == 1
    assert summary["by_brief_id"]["brief-a"][hold.VERDICT_BLOCKED] == 1
    assert summary["by_brief_id"]["brief-a"][hold.VERDICT_NOT_APPLICABLE] == 0
    assert summary["by_brief_id"]["brief-b"][hold.VERDICT_BLOCKED] == 1
    assert summary["by_brief_id"]["brief-b"][hold.VERDICT_NOT_APPLICABLE] == 1
    assert summary["pass_briefs"] == ["brief-a"]
    assert summary["pass_runs_total"] == 1
    assert summary["hold_threshold_met"] is False


def test_build_aggregate_summary_threshold_met_with_three_runs_two_briefs():
    reports = [
        _stub_report(brief_id="brief-a", run_label="run-1", verdict=hold.VERDICT_PASS),
        _stub_report(brief_id="brief-a", run_label="run-2", verdict=hold.VERDICT_PASS),
        _stub_report(brief_id="brief-b", run_label="run-1", verdict=hold.VERDICT_PASS),
    ]
    summary = hold.build_aggregate_summary(reports)
    assert summary["pass_runs_total"] == 3
    assert summary["pass_briefs"] == ["brief-a", "brief-b"]
    assert summary["hold_threshold_met"] is True


# ---------------------------------------------------------------------------
# 16-17. format_per_run_listing
# ---------------------------------------------------------------------------


def test_format_per_run_listing_includes_blocking_gate_for_blocked():
    reports = [
        _stub_report(brief_id="brief-a", run_label="run-1", verdict=hold.VERDICT_PASS),
        _stub_report(brief_id="brief-a", run_label="run-2", verdict=hold.VERDICT_BLOCKED),
        _stub_report(brief_id="brief-b", run_label="run-3", verdict=hold.VERDICT_NOT_APPLICABLE),
    ]
    out = hold.format_per_run_listing(reports, limit=20)
    assert "Per-run:" in out
    assert "brief-a/run-1" in out and "PASS" in out
    assert "brief-a/run-2" in out and "BLOCKED" in out
    assert "blocking gate: " + hold.GATE_FINAL_JUDGMENTS_ZERO_BYTES in out
    assert "brief-b/run-3" in out and "NOT_APPLICABLE" in out
    assert "reason: " + hold.GATE_FLAG_NEVER_ON in out
    # PASS rows use the - prefix; BLOCKED rows use the ! prefix.
    pass_lines = [
        line for line in out.splitlines() if line.startswith("- brief-a/run-1")
    ]
    blocked_lines = [
        line for line in out.splitlines() if line.startswith("! brief-a/run-2")
    ]
    assert len(pass_lines) == 1
    assert len(blocked_lines) == 1


def test_format_per_run_listing_respects_limit():
    reports = [
        _stub_report(
            brief_id="brief-a",
            run_label=f"run-{i}",
            verdict=hold.VERDICT_BLOCKED,
        )
        for i in range(5)
    ]
    out = hold.format_per_run_listing(reports, limit=2)
    listing_lines = [
        line for line in out.splitlines() if line.startswith("! brief-a/")
    ]
    assert len(listing_lines) == 2
    assert "and 3 more" in out

    out_no_limit = hold.format_per_run_listing(reports, limit=0)
    full_lines = [
        line for line in out_no_limit.splitlines() if line.startswith("! brief-a/")
    ]
    assert len(full_lines) == 5
    assert "and " not in out_no_limit.split("Per-run:", 1)[1]


# ---------------------------------------------------------------------------
# 18. write_report_json: protected path rejection + happy path
# ---------------------------------------------------------------------------


def test_write_report_json_rejects_protected_path_and_succeeds_outside(
    tmp_path, capsys
):
    summary = hold.build_aggregate_summary([])

    output_root = tmp_path / "output"
    (output_root / "exports").mkdir(parents=True)
    bad = output_root / "exports" / "report.json"

    with pytest.raises(SystemExit) as excinfo:
        hold.write_report_json([], summary, bad)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "protected" in err
    assert str(bad) in err
    assert not bad.exists()

    good = tmp_path / "report.json"
    hold.write_report_json([], summary, good)
    assert good.is_file()
    payload = json.loads(good.read_text(encoding="utf-8"))
    assert "summary" in payload and "runs" in payload
    assert payload["summary"]["total_runs_discovered"] == 0


# ---------------------------------------------------------------------------
# 19-22. run() end-to-end (synthetic tmp tree, runs_root injection)
# ---------------------------------------------------------------------------


def test_run_end_to_end_mixed_tree(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    pass_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=_full_pass_files(borderline_count=2),
    )
    blocked_files = _full_pass_files(borderline_count=2)
    blocked_files["final_judgments.jsonl"] = b""
    blocked_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-b",
        run_label="2026-04-21T00-00-00-000000+00-00__run-2",
        files=blocked_files,
    )
    notapp_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-c",
        run_label="2026-04-22T00-00-00-000000+00-00__run-3",
        files={
            "snippets.jsonl": b'{"k": 1}\n',
            "profile_summaries.jsonl": b'{"k": 2}\n',
            "final_judgments.jsonl": b'{"k": 3}\n',
            "facial_judgments.jsonl": (
                _facial_row(decision="FACIAL_YES", rationale="clean")
                + _facial_row(decision="FACIAL_NO", rationale="clean")
            ),
        },
    )

    rc = hold.run(_ns(limit=20), runs_root=runs_root)
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== Step B Hold Report ===" in out
    assert "total_runs_discovered      : 3" in out
    assert "PASS                       : 1" in out
    assert "BLOCKED                    : 1" in out
    assert "NOT_APPLICABLE             : 1" in out
    assert f"brief-a/{pass_dir.name}" in out and "PASS" in out
    assert f"brief-b/{blocked_dir.name}" in out and "BLOCKED" in out
    assert f"brief-c/{notapp_dir.name}" in out and "NOT_APPLICABLE" in out
    assert hold.GATE_FINAL_JUDGMENTS_ZERO_BYTES in out
    assert hold.GATE_FLAG_NEVER_ON in out
    assert "Note: 'PASS' means" in out


def test_run_report_out_writes_outside_output(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=_full_pass_files(borderline_count=2),
    )
    target = tmp_path / "hold.json"
    rc = hold.run(_ns(report_out=str(target)), runs_root=runs_root)
    assert rc == 0
    assert target.is_file()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["summary"]["total_runs_discovered"] == 1
    assert payload["summary"][hold.VERDICT_PASS] == 1
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["verdict"] == hold.VERDICT_PASS
    assert payload["runs"][0]["facial_borderline_count"] == 2
    # Runs serialize the run_dir as a string for portability.
    assert isinstance(payload["runs"][0]["run_dir"], str)


def test_run_empty_case(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    rc = hold.run(_ns(limit=20), runs_root=runs_root)
    assert rc == 0
    out = capsys.readouterr().out
    assert "No completed LinkedIn runs found" in out


def test_run_report_out_rejects_protected_path(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=_full_pass_files(borderline_count=2),
    )
    output_root = tmp_path / "output"
    (output_root / "exports").mkdir(parents=True)
    bad = output_root / "exports" / "foo.json"

    with pytest.raises(SystemExit) as excinfo:
        hold.run(_ns(report_out=str(bad)), runs_root=runs_root)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "protected" in err
    assert not bad.exists()


# ---------------------------------------------------------------------------
# 23. Real-repo smoke. Conditionally skipped when output/runs/linkedin is
# not populated (e.g. CI without the production output tree).
# ---------------------------------------------------------------------------


def _real_runs_root() -> Path:
    return PROJECT_ROOT / "output" / "runs" / "linkedin"


def _has_real_runs() -> bool:
    root = _real_runs_root()
    if not root.is_dir():
        return False
    for brief_dir in root.iterdir():
        if not brief_dir.is_dir():
            continue
        if brief_dir.name == "unknown":
            continue
        for run_dir in brief_dir.iterdir():
            if (
                run_dir.is_dir()
                and not run_dir.name.startswith("imported-")
                and (run_dir / "facial_judgments.jsonl").is_file()
            ):
                return True
    return False


@pytest.mark.skipif(
    not _has_real_runs(),
    reason=(
        "no real LinkedIn runs in output/runs/linkedin/ (skipped on CI / "
        "fresh checkouts)"
    ),
)
def test_classify_run_against_real_run_dir_does_not_crash():
    """Pin behavior on actual production data without writing anything.

    Per the spec: the hold tool must correctly classify the existing 6
    finalized runs. The spec preview predicted ``BLOCKED
    final_judgments_zero_bytes``, but the on-disk reality (verified by
    ripgrep over every facial_judgments.jsonl during implementation) is
    that NO existing run carries the ``[BORDERLINE\u2192YES alias]``
    canary or the ``facial_borderline_under_flag_off`` canary. So the
    correct verdict per the spec's own logic ordering is
    ``NOT_APPLICABLE`` (``flag_was_on=False``), NOT ``BLOCKED``. This
    test pins that behavior.
    """

    found = None
    for brief_dir in sorted(_real_runs_root().iterdir(), key=lambda p: p.name):
        if not brief_dir.is_dir() or brief_dir.name == "unknown":
            continue
        for run_dir in sorted(brief_dir.iterdir(), key=lambda p: p.name):
            if (
                run_dir.is_dir()
                and not run_dir.name.startswith("imported-")
                and (run_dir / "facial_judgments.jsonl").is_file()
            ):
                found = run_dir
                break
        if found is not None:
            break
    assert found is not None  # _has_real_runs guard already ensures this

    report = hold._classify_run(found)
    # Verdict is one of the three pinned values; classification did not
    # raise on real data.
    assert report.verdict in (
        hold.VERDICT_PASS,
        hold.VERDICT_BLOCKED,
        hold.VERDICT_NOT_APPLICABLE,
    )
    # Real-repo invariant pinned at slice-implementation time: with no
    # borderline canary on disk, flag_was_on is False and the verdict
    # is NOT_APPLICABLE. If this ever flips, a future run captured the
    # borderline canary and the spec's expected output flips with it.
    if report.facial_borderline_count == 0 and report.flag_off_canary_count == 0:
        assert report.flag_was_on is False
        assert report.verdict == hold.VERDICT_NOT_APPLICABLE
        assert report.blocking_gate == hold.GATE_FLAG_NEVER_ON


# ---------------------------------------------------------------------------
# 24. import-narrowness check (string-search on the new module's source)
# ---------------------------------------------------------------------------


class TestImportNarrowness:
    def test_tool_does_not_import_production_modules(self):
        text = Path(hold.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "from linkedin.",
            "from linkedin ",
            "from github.",
            "from github ",
            "from market_intelligence",
            "from shared.runtime_state",
            "from shared.external_evidence",
            "from shared.bias_controls",
            "from shared.judger",
            "from shared.schemas",
            "from shared.contracts",
            "import linkedin.",
            "import github.",
            "import market_intelligence",
        ):
            assert forbidden not in text, (
                f"step_b_hold_report must not import from forbidden module: "
                f"{forbidden}"
            )
