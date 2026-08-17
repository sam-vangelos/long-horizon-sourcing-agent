"""Tests for ``tools/discover_facial_gate_inputs.py`` (slice 11).

Hard rules pinned by these tests:

- No live calls. No imports from ``linkedin/``, ``github/``, or
  ``market_intelligence/`` (verified by string-search on the new module's
  source -- mirrors ``tests/test_facial_gate_experiment.py``).
- ``classify_run_inputs``, ``discover_run_dirs``, ``build_report``,
  ``format_report``, ``format_recommended_invocation`` are pure: same
  input -> same output, no clocks, no randomness.
- The default invocation writes nothing to disk; only ``--json-out`` writes,
  and it refuses paths under any protected ``output/`` subtree.
- Empty discovery exits 0 (discovery is a query, not a hard requirement).
- The four usability tiers (``usable_full``, ``usable_recovery_only``,
  ``usable_minimal``, ``incomplete_no_snippets``) are pinned by the
  classification tests; downstream consumers (the recommended-invocation
  formatter) read directly off this contract.
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

from tools import discover_facial_gate_inputs as disc  # noqa: E402


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

    ``files`` is ``{filename: contents_bytes}``. Missing keys are not
    created. The four-part path satisfies ``shared.output_paths.is_run_dir``
    when ``runs_root`` ends in ``output/runs``.
    """

    run_dir = runs_root / source / brief_id / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (files or {}).items():
        (run_dir / name).write_bytes(payload)
    return run_dir


def _runs_root(tmp_path: Path) -> Path:
    """Return a path that ``is_run_dir`` accepts as an ``output/runs/...``.

    ``shared.output_paths._parts_after_output`` walks parents looking for a
    component literally named ``"output"`` and treats anything four-deep
    after that as a run-dir. We anchor on ``tmp_path/output/runs``.
    """

    root = tmp_path / "output" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ns(**overrides) -> argparse.Namespace:
    """Build a defaults-aligned argparse namespace for ``run()``."""

    base = dict(
        source="linkedin",
        brief_id=None,
        require_finals=False,
        require_summaries=False,
        limit=20,
        json_out=None,
        include_legacy=False,
        include_unknown_brief=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# 1-4. classify_run_inputs
# ---------------------------------------------------------------------------


def test_classify_run_inputs_usable_full(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="3000000006",
        run_label="2026-04-14T10-24-45-885842+00-00__run-14",
        files={
            "snippets.jsonl": b'{"hi": 1}\n',
            "profile_summaries.jsonl": b'{"hi": 2}\n',
            "final_judgments.jsonl": b'{"hi": 3}\n',
        },
    )
    c = disc.classify_run_inputs(run_dir)
    assert c.usability == disc.USABILITY_FULL
    assert c.has_snippets is True
    assert c.has_profile_summaries is True
    assert c.has_final_judgments is True
    assert c.snippets_size_bytes is not None and c.snippets_size_bytes > 0
    assert c.summaries_size_bytes is not None and c.summaries_size_bytes > 0
    assert c.finals_size_bytes is not None and c.finals_size_bytes > 0
    assert c.incomplete_reason == ""
    assert c.brief_id == "3000000006"
    assert c.source == "linkedin"
    assert c.run_label == "2026-04-14T10-24-45-885842+00-00__run-14"


def test_classify_run_inputs_recovery_only(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-x",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={
            "snippets.jsonl": b'{"k": 1}\n',
            "final_judgments.jsonl": b'{"k": 3}\n',
        },
    )
    c = disc.classify_run_inputs(run_dir)
    assert c.usability == disc.USABILITY_RECOVERY_ONLY
    assert c.has_snippets is True
    assert c.has_profile_summaries is False
    assert c.has_final_judgments is True
    assert c.summaries_size_bytes is None
    assert c.incomplete_reason == ""


def test_classify_run_inputs_minimal(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-y",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b'{"k": 1}\n'},
    )
    c = disc.classify_run_inputs(run_dir)
    assert c.usability == disc.USABILITY_MINIMAL
    assert c.has_snippets is True
    assert c.has_profile_summaries is False
    assert c.has_final_judgments is False
    assert c.summaries_size_bytes is None
    assert c.finals_size_bytes is None
    assert c.incomplete_reason == ""


def test_classify_run_inputs_incomplete_no_snippets(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir_no_files = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-z",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files=None,
    )
    c = disc.classify_run_inputs(run_dir_no_files)
    assert c.usability == disc.USABILITY_INCOMPLETE
    assert c.has_snippets is False
    assert "snippets.jsonl" in c.incomplete_reason

    run_dir_finals_only = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-z",
        run_label="2026-04-21T00-00-00-000000+00-00__run-2",
        files={"final_judgments.jsonl": b'{"k": 1}\n'},
    )
    c2 = disc.classify_run_inputs(run_dir_finals_only)
    assert c2.usability == disc.USABILITY_INCOMPLETE
    assert c2.has_snippets is False
    assert c2.has_final_judgments is True
    assert "snippets.jsonl" in c2.incomplete_reason


# ---------------------------------------------------------------------------
# 5-8. discover_run_dirs
# ---------------------------------------------------------------------------


def test_discover_run_dirs_finds_run_shaped_dirs_only(tmp_path):
    runs_root = _runs_root(tmp_path)
    real_run = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )
    # A bare brief dir (parent of run-dirs) -- not a run-dir, must NOT show up.
    (runs_root / "linkedin" / "brief-a-empty").mkdir(parents=True)
    # A loose file inside a brief dir -- must NOT show up either.
    (runs_root / "linkedin" / "brief-a" / "stray.txt").write_text("nope")

    found = disc.discover_run_dirs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=False,
        include_unknown_brief=False,
        brief_id_filter=None,
    )
    assert real_run in found
    assert all(p.is_dir() for p in found)
    assert all(p != runs_root / "linkedin" / "brief-a" for p in found)
    assert all(
        p != runs_root / "linkedin" / "brief-a" / "stray.txt" for p in found
    )


def test_discover_run_dirs_skips_legacy_by_default(tmp_path):
    runs_root = _runs_root(tmp_path)
    legacy_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="imported-2026-04-08T22-44-23-171843+00-00__legacy-1",
        files={"snippets.jsonl": b"{}\n"},
    )
    real_run = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )

    without = disc.discover_run_dirs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=False,
        include_unknown_brief=False,
        brief_id_filter=None,
    )
    assert real_run in without
    assert legacy_dir not in without

    with_legacy = disc.discover_run_dirs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=True,
        include_unknown_brief=False,
        brief_id_filter=None,
    )
    assert real_run in with_legacy
    assert legacy_dir in with_legacy


def test_discover_run_dirs_skips_unknown_brief_by_default(tmp_path):
    runs_root = _runs_root(tmp_path)
    real_run = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )
    unknown_run = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="unknown",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )

    without = disc.discover_run_dirs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=False,
        include_unknown_brief=False,
        brief_id_filter=None,
    )
    assert real_run in without
    assert unknown_run not in without

    with_unknown = disc.discover_run_dirs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=False,
        include_unknown_brief=True,
        brief_id_filter=None,
    )
    assert real_run in with_unknown
    assert unknown_run in with_unknown


def test_discover_run_dirs_brief_id_filter(tmp_path):
    runs_root = _runs_root(tmp_path)
    matching = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="3000000006",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )
    other = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="3000000007",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )

    found = disc.discover_run_dirs(
        source="linkedin",
        runs_root=runs_root,
        include_legacy=False,
        include_unknown_brief=False,
        brief_id_filter="3000000006",
    )
    assert matching in found
    assert other not in found


# ---------------------------------------------------------------------------
# 9. build_report aggregation
# ---------------------------------------------------------------------------


def test_build_report_aggregation_counts_and_by_brief(tmp_path):
    runs_root = _runs_root(tmp_path)

    full = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={
            "snippets.jsonl": b"{}\n",
            "profile_summaries.jsonl": b"{}\n",
            "final_judgments.jsonl": b"{}\n",
        },
    )
    recovery = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-21T00-00-00-000000+00-00__run-2",
        files={
            "snippets.jsonl": b"{}\n",
            "final_judgments.jsonl": b"{}\n",
        },
    )
    minimal = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-b",
        run_label="2026-04-22T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )
    incomplete = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-b",
        run_label="2026-04-22T00-00-00-000000+00-00__run-2",
        files={"final_judgments.jsonl": b"{}\n"},
    )

    classifications = [
        disc.classify_run_inputs(d)
        for d in (full, recovery, minimal, incomplete)
    ]
    report = disc.build_report(classifications)

    assert report["total_runs_discovered"] == 4
    assert report["usable_full"] == 1
    assert report["usable_recovery_only"] == 1
    assert report["usable_minimal"] == 1
    assert report["incomplete_no_snippets"] == 1
    assert set(report["by_brief_id"].keys()) == {"brief-a", "brief-b"}
    assert report["by_brief_id"]["brief-a"]["usable_full"] == 1
    assert report["by_brief_id"]["brief-a"]["usable_recovery_only"] == 1
    assert report["by_brief_id"]["brief-a"]["usable_minimal"] == 0
    assert report["by_brief_id"]["brief-b"]["usable_minimal"] == 1
    assert report["by_brief_id"]["brief-b"]["incomplete_no_snippets"] == 1


# ---------------------------------------------------------------------------
# 10-11. format_report
# ---------------------------------------------------------------------------


def test_format_report_human_output_respects_limit(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dirs = []
    for i in range(5):
        run_dirs.append(
            _make_run_tree(
                runs_root=runs_root,
                source="linkedin",
                brief_id="brief-a",
                run_label=f"2026-04-20T00-00-{i:02d}-000000+00-00__run-{i + 1}",
                files={
                    "snippets.jsonl": b"{}\n",
                    "profile_summaries.jsonl": b"{}\n",
                    "final_judgments.jsonl": b"{}\n",
                },
            )
        )
    classifications = [disc.classify_run_inputs(d) for d in run_dirs]
    report = disc.build_report(classifications)

    out = disc.format_report(report, limit=2)
    assert "=== Facial-Gate Input Discovery ===" in out
    assert "total_runs_discovered     : 5" in out
    assert "usable_full" in out
    assert "By brief-id:" in out
    assert "brief-a" in out
    # Exactly two run lines plus the truncation marker.
    listing_lines = [
        line for line in out.splitlines() if line.startswith("- brief-a/")
    ]
    assert len(listing_lines) == 2
    assert "and 3 more" in out

    out_no_limit = disc.format_report(report, limit=0)
    listing_lines_full = [
        line for line in out_no_limit.splitlines() if line.startswith("- brief-a/")
    ]
    assert len(listing_lines_full) == 5
    assert "and " not in out_no_limit.split("Runs:", 1)[1]


def test_format_report_empty_case_renders_zero_total():
    empty = disc.build_report([])
    out = disc.format_report(empty, limit=20)
    assert "total_runs_discovered     : 0" in out
    assert "(no run-dirs discovered)" in out
    assert "Recommended next step" not in out
    assert "By brief-id:" in out
    assert "(no briefs)" in out


# ---------------------------------------------------------------------------
# 12-15. format_recommended_invocation
# ---------------------------------------------------------------------------


def _make_classification(
    *, run_dir: Path, usability: str
) -> disc.RunInputClassification:
    return disc.RunInputClassification(
        source="linkedin",
        brief_id=run_dir.parent.name,
        run_dir=run_dir,
        run_label=run_dir.name,
        has_snippets=True,
        has_profile_summaries=usability == disc.USABILITY_FULL,
        has_final_judgments=usability != disc.USABILITY_MINIMAL,
        snippets_size_bytes=10,
        summaries_size_bytes=10 if usability == disc.USABILITY_FULL else None,
        finals_size_bytes=10 if usability != disc.USABILITY_MINIMAL else None,
        usability=usability,
        incomplete_reason="",
    )


def test_format_recommended_invocation_usable_full(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b""},
    )
    c = _make_classification(run_dir=run_dir, usability=disc.USABILITY_FULL)
    block = disc.format_recommended_invocation(c)
    assert "Recommended next step" in block
    assert "facial_gate_experiment.py" in block
    assert "--snippets" in block
    assert "--profile-summaries" in block
    assert "--final-judgments" in block
    assert "recommend_facial_gate.py" in block
    assert "quality-degraded" not in block


def test_format_recommended_invocation_recovery_only(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b""},
    )
    c = _make_classification(
        run_dir=run_dir, usability=disc.USABILITY_RECOVERY_ONLY
    )
    block = disc.format_recommended_invocation(c)
    assert "--profile-summaries" not in block
    assert "--final-judgments" in block
    assert "facial_gate_experiment.py" in block
    assert "quality-degraded" not in block


def test_format_recommended_invocation_minimal(tmp_path):
    runs_root = _runs_root(tmp_path)
    run_dir = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b""},
    )
    c = _make_classification(run_dir=run_dir, usability=disc.USABILITY_MINIMAL)
    block = disc.format_recommended_invocation(c)
    assert "--profile-summaries" not in block
    assert "--final-judgments" not in block
    assert "--snippets" in block
    assert "quality-degraded" in block


def test_format_recommended_invocation_zero_usable():
    block = disc.format_recommended_invocation(None)
    assert "No usable runs found" in block
    assert "snippets.jsonl" in block
    assert "linkedin/session_orchestrator.py" in block


# ---------------------------------------------------------------------------
# 16. write_report_json: protected path rejection + happy path
# ---------------------------------------------------------------------------


def test_write_report_json_rejects_protected_path_and_succeeds_outside(
    tmp_path, capsys
):
    report = {"total_runs_discovered": 0}

    output_root = tmp_path / "output"
    (output_root / "exports").mkdir(parents=True)
    bad_path = output_root / "exports" / "report.json"

    with pytest.raises(SystemExit) as excinfo:
        disc.write_report_json(report, bad_path)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "protected" in err
    assert str(bad_path) in err
    assert not bad_path.exists()

    good_path = tmp_path / "report.json"
    disc.write_report_json(report, good_path)
    assert good_path.is_file()
    parsed = json.loads(good_path.read_text(encoding="utf-8"))
    assert parsed == report


# ---------------------------------------------------------------------------
# 17-20. run() end-to-end (synthetic tmp tree, runs_root injection)
# ---------------------------------------------------------------------------


def test_run_end_to_end_mixed_tree(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    full = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={
            "snippets.jsonl": b"{}\n",
            "profile_summaries.jsonl": b"{}\n",
            "final_judgments.jsonl": b"{}\n",
        },
    )
    incomplete = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-b",
        run_label="2026-04-22T00-00-00-000000+00-00__run-2",
        files={"final_judgments.jsonl": b"{}\n"},
    )

    rc = disc.run(_ns(limit=20), runs_root=runs_root)
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_runs_discovered     : 2" in out
    assert "usable_full" in out
    assert "incomplete_no_snippets" in out
    assert f"- brief-a/{full.name}" in out
    assert f"! brief-b/{incomplete.name}" in out
    assert "Recommended next step" in out
    assert "--profile-summaries" in out


def test_run_empty_case(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    rc = disc.run(_ns(limit=20), runs_root=runs_root)
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_runs_discovered     : 0" in out
    assert "No usable runs found" in out
    assert "linkedin/session_orchestrator.py" in out


def test_run_brief_id_filter_smoke(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="3000000006",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={
            "snippets.jsonl": b"{}\n",
            "profile_summaries.jsonl": b"{}\n",
            "final_judgments.jsonl": b"{}\n",
        },
    )
    other = _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="3000000007",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={
            "snippets.jsonl": b"{}\n",
            "profile_summaries.jsonl": b"{}\n",
            "final_judgments.jsonl": b"{}\n",
        },
    )
    rc = disc.run(_ns(brief_id="3000000006"), runs_root=runs_root)
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_runs_discovered     : 1" in out
    assert "3000000006" in out
    assert "3000000007" not in out.split("Recommended next step", 1)[0]
    # Sanity: ``other`` run-dir did get created (we just filtered it out).
    assert other.is_dir()


def test_run_json_out_writes_outside_output(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={
            "snippets.jsonl": b"{}\n",
            "profile_summaries.jsonl": b"{}\n",
            "final_judgments.jsonl": b"{}\n",
        },
    )
    target = tmp_path / "discover.json"
    rc = disc.run(_ns(json_out=str(target)), runs_root=runs_root)
    assert rc == 0
    assert target.is_file()
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed["total_runs_discovered"] == 1
    assert parsed["usable_full"] == 1
    assert "brief-a" in parsed["by_brief_id"]


def test_run_json_out_rejects_protected_path(tmp_path, capsys):
    runs_root = _runs_root(tmp_path)
    _make_run_tree(
        runs_root=runs_root,
        source="linkedin",
        brief_id="brief-a",
        run_label="2026-04-20T00-00-00-000000+00-00__run-1",
        files={"snippets.jsonl": b"{}\n"},
    )
    output_root = tmp_path / "output"
    (output_root / "exports").mkdir(parents=True)
    bad = output_root / "exports" / "foo.json"

    with pytest.raises(SystemExit) as excinfo:
        disc.run(_ns(json_out=str(bad)), runs_root=runs_root)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "protected" in err
    assert not bad.exists()


# ---------------------------------------------------------------------------
# 21. import-narrowness check (string-search on the new module's source)
# ---------------------------------------------------------------------------


class TestImportNarrowness:
    def test_tool_does_not_import_production_modules(self):
        text = Path(disc.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "from linkedin.",
            "from linkedin ",
            "from github.",
            "from github ",
            "from market_intelligence",
            "from shared.runtime_state",
            "from shared.external_evidence",
            "from shared.judger",
            "from shared.schemas",
            "import linkedin.",
            "import github.",
            "import market_intelligence",
        ):
            assert forbidden not in text, (
                f"discover tool must not import from forbidden module: "
                f"{forbidden}"
            )
