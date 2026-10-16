"""Tests for ``tools/aggregate_shadow_judgments.py`` (slice 4).

Hard rules pinned by these tests:

- No live calls. No imports from ``linkedin/``, ``github/``,
  ``market_intelligence/``, or ``shared/external_evidence/provider.py``.
- ``aggregate`` is a pure function: same input -> same output, no I/O,
  no clocks.
- A malformed JSONL line is counted as a parse failure, not a crash.
- The default invocation writes nothing to disk; only ``--json-out`` writes.
- ``--changed-only`` and ``--save-flips-only`` are mutually exclusive.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import aggregate_shadow_judgments as agg  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers (small, in-line, mirror the on-disk schema in shadow_writer)
# ---------------------------------------------------------------------------


def _row(
    *,
    candidate_name: str = "Alice Liddell",
    profile_url: str = "https://linkedin.com/in/alice",
    source_string_id: int = 1,
    page: int = 1,
    result_rank: int = 1,
    trigger_reason: str = "academic_context",
    external_evidence_status: str = "evidence_present",
    identity_confidence=0.8,
    evidence_refs_count: int = 2,
    baseline=None,
    enriched=None,
    diff=None,
    timestamp: str = "2026-04-25T00:00:00Z",
    feature_version: str = "slice2",
) -> dict:
    if baseline is None:
        baseline = {
            "decision": "REJECT",
            "path": "core",
            "rationale": "baseline rationale",
            "confidence": 0.7,
        }
    if enriched is None:
        enriched = {
            "decision": "SAVE",
            "path": "core",
            "rationale": "enriched rationale",
            "confidence": 0.93,
        }
    if diff is None:
        diff = _diff(
            baseline=baseline,
            enriched=enriched,
            decision_changed=baseline["decision"] != enriched["decision"],
            path_changed=baseline["path"] != enriched["path"],
            rationale_changed=baseline["rationale"] != enriched["rationale"],
        )
    return {
        "candidate_name": candidate_name,
        "profile_url": profile_url,
        "source_string_id": source_string_id,
        "page": page,
        "result_rank": result_rank,
        "trigger_reason": trigger_reason,
        "external_evidence_status": external_evidence_status,
        "identity_confidence": identity_confidence,
        "evidence_refs_count": evidence_refs_count,
        "baseline": baseline,
        "enriched": enriched,
        "diff": diff,
        "timestamp": timestamp,
        "feature_version": feature_version,
    }


def _diff(
    *,
    baseline: dict,
    enriched: dict,
    decision_changed: bool,
    path_changed: bool,
    rationale_changed: bool,
) -> dict:
    return {
        "computed": True,
        "decision_changed": decision_changed,
        "decision_baseline": baseline["decision"],
        "decision_enriched": enriched["decision"],
        "path_changed": path_changed,
        "path_baseline": baseline["path"],
        "path_enriched": enriched["path"],
        "rationale_changed": rationale_changed,
        "confidence_delta": float(enriched["confidence"]) - float(baseline["confidence"]),
    }


def _records_from(rows: list[dict]):
    for row in rows:
        yield row, None


# ---------------------------------------------------------------------------
# aggregate: pure function tests
# ---------------------------------------------------------------------------


def test_aggregate_empty_iterator_all_zeros():
    summary = agg.aggregate(iter([]))
    assert summary["total_rows_read"] == 0
    assert summary["total_parse_failures"] == 0
    assert summary["total_compared"] == 0
    assert summary["same_decision"] == 0
    assert summary["decision_changed"] == 0
    assert summary["reject_to_save"] == 0
    assert summary["save_to_reject"] == 0
    assert summary["confidence_delta_avg"] == 0.0
    assert summary["confidence_delta_abs_avg"] == 0.0
    assert summary["evidence_refs_count_avg"] == 0.0
    assert summary["identity_confidence_avg"] == 0.0
    assert summary["gate_trigger_breakdown"] == {}
    assert summary["other_statuses"] == {}


def test_aggregate_evidence_present_same_decision():
    baseline = {
        "decision": "REJECT",
        "path": "core",
        "rationale": "same",
        "confidence": 0.6,
    }
    enriched = {
        "decision": "REJECT",
        "path": "core",
        "rationale": "same",
        "confidence": 0.7,
    }
    row = _row(
        baseline=baseline,
        enriched=enriched,
        diff=_diff(
            baseline=baseline,
            enriched=enriched,
            decision_changed=False,
            path_changed=False,
            rationale_changed=False,
        ),
    )
    summary = agg.aggregate(_records_from([row]))
    assert summary["total_compared"] == 1
    assert summary["same_decision"] == 1
    assert summary["decision_changed"] == 0
    assert summary["evidence_present_total"] == 1


def test_aggregate_reject_to_save_flip():
    summary = agg.aggregate(_records_from([_row()]))
    assert summary["reject_to_save"] == 1
    assert summary["decision_changed"] == 1
    assert summary["save_to_reject"] == 0
    assert summary["total_compared"] == 1


def test_aggregate_reject_to_inferential_save_counts_as_flip():
    baseline = {
        "decision": "REJECT",
        "path": "core",
        "rationale": "r",
        "confidence": 0.5,
    }
    enriched = {
        "decision": "INFERENTIAL_SAVE",
        "path": "core",
        "rationale": "r",
        "confidence": 0.7,
    }
    row = _row(
        baseline=baseline,
        enriched=enriched,
        diff=_diff(
            baseline=baseline,
            enriched=enriched,
            decision_changed=True,
            path_changed=False,
            rationale_changed=False,
        ),
    )
    summary = agg.aggregate(_records_from([row]))
    assert summary["reject_to_save"] == 1
    assert summary["save_to_reject"] == 0


def test_aggregate_save_to_reject_flip():
    baseline = {
        "decision": "SAVE",
        "path": "core",
        "rationale": "r",
        "confidence": 0.9,
    }
    enriched = {
        "decision": "REJECT",
        "path": "core",
        "rationale": "r",
        "confidence": 0.4,
    }
    row = _row(
        baseline=baseline,
        enriched=enriched,
        diff=_diff(
            baseline=baseline,
            enriched=enriched,
            decision_changed=True,
            path_changed=False,
            rationale_changed=False,
        ),
    )
    summary = agg.aggregate(_records_from([row]))
    assert summary["save_to_reject"] == 1
    assert summary["reject_to_save"] == 0


def test_aggregate_path_only_change():
    baseline = {
        "decision": "SAVE",
        "path": "core",
        "rationale": "same",
        "confidence": 0.8,
    }
    enriched = {
        "decision": "SAVE",
        "path": "transferable",
        "rationale": "same",
        "confidence": 0.85,
    }
    row = _row(
        baseline=baseline,
        enriched=enriched,
        diff=_diff(
            baseline=baseline,
            enriched=enriched,
            decision_changed=False,
            path_changed=True,
            rationale_changed=False,
        ),
    )
    summary = agg.aggregate(_records_from([row]))
    assert summary["path_only_changes"] == 1
    assert summary["same_decision"] == 1
    assert summary["decision_changed"] == 0
    assert summary["rationale_only_changes"] == 0


def test_aggregate_rationale_only_change():
    baseline = {
        "decision": "SAVE",
        "path": "core",
        "rationale": "v1",
        "confidence": 0.8,
    }
    enriched = {
        "decision": "SAVE",
        "path": "core",
        "rationale": "v2",
        "confidence": 0.85,
    }
    row = _row(
        baseline=baseline,
        enriched=enriched,
        diff=_diff(
            baseline=baseline,
            enriched=enriched,
            decision_changed=False,
            path_changed=False,
            rationale_changed=True,
        ),
    )
    summary = agg.aggregate(_records_from([row]))
    assert summary["rationale_only_changes"] == 1
    assert summary["path_only_changes"] == 0


def test_aggregate_buckets_failure_statuses_and_other():
    statuses = [
        "weak_citations",
        "quota_exhausted",
        "timeout",
        "parse_failure",
        "disabled_no_api_key",
        "disabled_by_config",
        "skipped_no_trigger",
        "galaxy_brained",
    ]
    rows: list[dict] = []
    for status in statuses:
        rows.append(
            _row(
                external_evidence_status=status,
                enriched=None,
                diff={"computed": False, "reason": status},
            )
        )
    summary = agg.aggregate(_records_from(rows))
    assert summary["weak_citations"] == 1
    assert summary["quota_exhausted"] == 1
    assert summary["timeout"] == 1
    assert summary["parse_failure_provider"] == 1
    assert summary["disabled_no_api_key"] == 1
    assert summary["disabled_by_config"] == 1
    assert summary["skipped_no_trigger"] == 1
    assert summary["other_statuses"] == {"galaxy_brained": 1}
    # All these rows have diff.computed == False, so unavailable_external_evidence
    # equals the row count.
    assert summary["unavailable_external_evidence"] == len(rows)
    # None of them count as compared.
    assert summary["total_compared"] == 0
    assert summary["same_decision"] == 0
    assert summary["decision_changed"] == 0


def test_aggregate_unavailable_when_enriched_none():
    row = _row(
        external_evidence_status="evidence_present",
        enriched=None,
        diff={"computed": False, "reason": "no_enriched_decision"},
    )
    summary = agg.aggregate(_records_from([row]))
    assert summary["unavailable_external_evidence"] == 1
    assert summary["same_decision"] == 0
    assert summary["decision_changed"] == 0
    assert summary["total_compared"] == 0


def test_aggregate_gate_trigger_breakdown():
    rows = [
        _row(trigger_reason="academic_context"),
        _row(trigger_reason="academic_context"),
        _row(trigger_reason="sparse_profile"),
        _row(trigger_reason=""),
    ]
    summary = agg.aggregate(_records_from(rows))
    assert summary["gate_trigger_breakdown"] == {
        "": 1,
        "academic_context": 2,
        "sparse_profile": 1,
    }


def test_aggregate_evidence_refs_avg_only_over_evidence_present():
    rows = [
        _row(external_evidence_status="evidence_present", evidence_refs_count=2),
        _row(external_evidence_status="evidence_present", evidence_refs_count=4),
        _row(
            external_evidence_status="weak_citations",
            evidence_refs_count=999,  # must be ignored
            enriched=None,
            diff={"computed": False, "reason": "weak_citations"},
        ),
    ]
    summary = agg.aggregate(_records_from(rows))
    assert summary["evidence_refs_count_avg"] == pytest.approx(3.0)


def test_aggregate_identity_confidence_avg_skips_none():
    rows = [
        _row(identity_confidence=0.4),
        _row(identity_confidence=None),
        _row(identity_confidence=0.8),
    ]
    summary = agg.aggregate(_records_from(rows))
    assert summary["identity_confidence_avg"] == pytest.approx(0.6)


def test_aggregate_confidence_delta_averages_over_compared_only():
    # Two compared rows with deltas +0.1 and -0.3; one unavailable row.
    baseline_a = {"decision": "REJECT", "path": "core", "rationale": "a", "confidence": 0.5}
    enriched_a = {"decision": "REJECT", "path": "core", "rationale": "a", "confidence": 0.6}
    baseline_b = {"decision": "SAVE", "path": "core", "rationale": "b", "confidence": 0.9}
    enriched_b = {"decision": "SAVE", "path": "core", "rationale": "b", "confidence": 0.6}
    rows = [
        _row(
            baseline=baseline_a,
            enriched=enriched_a,
            diff=_diff(
                baseline=baseline_a,
                enriched=enriched_a,
                decision_changed=False,
                path_changed=False,
                rationale_changed=False,
            ),
        ),
        _row(
            baseline=baseline_b,
            enriched=enriched_b,
            diff=_diff(
                baseline=baseline_b,
                enriched=enriched_b,
                decision_changed=False,
                path_changed=False,
                rationale_changed=False,
            ),
        ),
        _row(
            external_evidence_status="quota_exhausted",
            enriched=None,
            diff={"computed": False, "reason": "quota_exhausted"},
        ),
    ]
    summary = agg.aggregate(_records_from(rows))
    assert summary["total_compared"] == 2
    assert summary["confidence_delta_avg"] == pytest.approx(-0.1)
    assert summary["confidence_delta_abs_avg"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# iter_records
# ---------------------------------------------------------------------------


def test_iter_records_happy_path_two_lines():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "shadow_final_judgments.jsonl"
        p.write_text(
            json.dumps(_row(candidate_name="A")) + "\n"
            + json.dumps(_row(candidate_name="B")) + "\n",
            encoding="utf-8",
        )
        records = list(agg.iter_records([p]))
        assert len(records) == 2
        assert all(parsed is not None for parsed, _ in records)
        names = [parsed["candidate_name"] for parsed, _ in records]
        assert names == ["A", "B"]


def test_iter_records_malformed_line_is_parse_failure():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "shadow_final_judgments.jsonl"
        p.write_text(
            json.dumps(_row(candidate_name="A")) + "\n"
            + "not json\n"
            + json.dumps(_row(candidate_name="B")) + "\n",
            encoding="utf-8",
        )
        summary = agg.aggregate(agg.iter_records([p]))
        assert summary["total_rows_read"] == 2
        assert summary["total_parse_failures"] == 1


def test_iter_records_blank_lines_are_skipped_silently():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "shadow_final_judgments.jsonl"
        p.write_text(
            "\n"
            + json.dumps(_row(candidate_name="A")) + "\n"
            + "   \n"
            + json.dumps(_row(candidate_name="B")) + "\n",
            encoding="utf-8",
        )
        summary = agg.aggregate(agg.iter_records([p]))
        assert summary["total_rows_read"] == 2
        assert summary["total_parse_failures"] == 0


def test_iter_records_non_dict_top_level_is_parse_failure():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "shadow_final_judgments.jsonl"
        p.write_text(
            json.dumps(_row(candidate_name="A")) + "\n"
            + "[1, 2, 3]\n",
            encoding="utf-8",
        )
        summary = agg.aggregate(agg.iter_records([p]))
        assert summary["total_rows_read"] == 1
        assert summary["total_parse_failures"] == 1


# ---------------------------------------------------------------------------
# select_changed_rows
# ---------------------------------------------------------------------------


def _row_with_delta(name: str, delta: float, *, decision_changed: bool = True) -> dict:
    baseline = {
        "decision": "REJECT" if decision_changed else "SAVE",
        "path": "core",
        "rationale": "r",
        "confidence": 0.5,
    }
    enriched = {
        "decision": "SAVE" if decision_changed else "SAVE",
        "path": "core",
        "rationale": "r",
        "confidence": 0.5 + delta,
    }
    return _row(
        candidate_name=name,
        baseline=baseline,
        enriched=enriched,
        diff={
            "computed": True,
            "decision_changed": decision_changed,
            "decision_baseline": baseline["decision"],
            "decision_enriched": enriched["decision"],
            "path_changed": False,
            "path_baseline": baseline["path"],
            "path_enriched": enriched["path"],
            "rationale_changed": False,
            "confidence_delta": delta,
        },
    )


def test_select_changed_rows_orders_by_abs_delta_desc():
    rows = [
        _row_with_delta("Small", 0.05),
        _row_with_delta("Big", -0.40),
        _row_with_delta("Mid", 0.20),
    ]
    out = agg.select_changed_rows(rows, save_flips_only=False)
    deltas = [r["diff"]["confidence_delta"] for r in out]
    assert deltas == [-0.40, 0.20, 0.05]


def test_select_changed_rows_save_flips_only_excludes_path_and_rationale():
    # Path-only change.
    baseline = {"decision": "SAVE", "path": "core", "rationale": "r", "confidence": 0.8}
    enriched_path = {"decision": "SAVE", "path": "transferable", "rationale": "r", "confidence": 0.85}
    path_only = _row(
        candidate_name="PathOnly",
        baseline=baseline,
        enriched=enriched_path,
        diff=_diff(
            baseline=baseline,
            enriched=enriched_path,
            decision_changed=False,
            path_changed=True,
            rationale_changed=False,
        ),
    )
    # Rationale-only change.
    enriched_rat = {"decision": "SAVE", "path": "core", "rationale": "r2", "confidence": 0.81}
    rationale_only = _row(
        candidate_name="RationaleOnly",
        baseline=baseline,
        enriched=enriched_rat,
        diff=_diff(
            baseline=baseline,
            enriched=enriched_rat,
            decision_changed=False,
            path_changed=False,
            rationale_changed=True,
        ),
    )
    # Real flip.
    flip = _row_with_delta("Flip", 0.30, decision_changed=True)

    out = agg.select_changed_rows(
        [path_only, rationale_only, flip],
        save_flips_only=True,
    )
    names = [r["candidate_name"] for r in out]
    assert names == ["Flip"]


def test_select_changed_rows_excludes_diff_not_computed():
    not_computed = _row(
        external_evidence_status="quota_exhausted",
        enriched=None,
        diff={"computed": False, "reason": "quota_exhausted"},
    )
    flip = _row_with_delta("Flip", 0.10, decision_changed=True)
    out = agg.select_changed_rows([not_computed, flip], save_flips_only=False)
    assert [r["candidate_name"] for r in out] == ["Flip"]


def test_select_changed_rows_stable_secondary_sort():
    a = _row_with_delta("Alpha", 0.10)
    b = _row_with_delta("Bravo", 0.10)
    out = agg.select_changed_rows([b, a], save_flips_only=False)
    # Equal abs(delta) -> sort by name ascending.
    assert [r["candidate_name"] for r in out] == ["Alpha", "Bravo"]


# ---------------------------------------------------------------------------
# resolve_input_paths
# ---------------------------------------------------------------------------


def test_resolve_input_paths_explicit_dedup_and_missing_filter():
    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a.jsonl"
        b = Path(td) / "b.jsonl"
        a.write_text("", encoding="utf-8")
        b.write_text("", encoding="utf-8")
        missing = Path(td) / "missing.jsonl"
        out = agg.resolve_input_paths(
            [str(a), str(missing), str(b), str(a)],
            None,
        )
        # Order preserved (a first, b second), missing dropped, dup dropped.
        assert [p.name for p in out] == ["a.jsonl", "b.jsonl"]


def test_resolve_input_paths_glob_and_dedup_with_explicit():
    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "shadow_final_judgments.jsonl"
        sub = Path(td) / "sub"
        sub.mkdir()
        b = sub / "shadow_final_judgments.jsonl"
        a.write_text("", encoding="utf-8")
        b.write_text("", encoding="utf-8")
        pattern = str(Path(td) / "**" / "shadow_final_judgments.jsonl")
        out = agg.resolve_input_paths([str(a)], pattern)
        # ``a`` is hit by both explicit and glob; should appear once. ``b`` is
        # only hit by the glob.
        names = sorted(p.parent.name + "/" + p.name for p in out)
        assert names == sorted(
            [
                f"{Path(td).name}/shadow_final_judgments.jsonl",
                "sub/shadow_final_judgments.jsonl",
            ]
        )


def test_resolve_input_paths_empty_returns_empty():
    out = agg.resolve_input_paths([], None)
    assert out == []


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def test_format_summary_contains_required_labels():
    summary = agg._empty_summary()
    summary["total_rows_read"] = 5
    summary["total_compared"] = 3
    summary["decision_changed"] = 1
    summary["reject_to_save"] = 1
    summary["save_to_reject"] = 0
    summary["unavailable_external_evidence"] = 2
    summary["weak_citations"] = 1
    summary["gate_trigger_breakdown"] = {"academic_context": 2}

    text = agg.format_summary(summary)
    for label in (
        "total_compared",
        "decision_changed",
        "reject_to_save",
        "save_to_reject",
        "unavailable_external_evidence",
        "weak_citations",
        "gate_trigger_breakdown",
    ):
        assert label in text


def test_format_changed_rows_renders_known_fixture_shape():
    row = _row(
        candidate_name="Alice",
        evidence_refs_count=2,
        trigger_reason="academic_context",
        baseline={"decision": "REJECT", "path": "core", "rationale": "r", "confidence": 0.5},
        enriched={"decision": "SAVE", "path": "core", "rationale": "r2", "confidence": 0.73},
        diff={
            "computed": True,
            "decision_changed": True,
            "decision_baseline": "REJECT",
            "decision_enriched": "SAVE",
            "path_changed": False,
            "path_baseline": "core",
            "path_enriched": "core",
            "rationale_changed": True,
            "confidence_delta": 0.23,
        },
    )
    text = agg.format_changed_rows([row], limit=10)
    assert "REJECT -> SAVE" in text
    assert "+0.230" in text
    assert "refs=2" in text
    assert "trigger=academic_context" in text


def test_format_changed_rows_truncation():
    rows = [
        _row_with_delta(f"C{i:02d}", 0.10 + i * 0.001) for i in range(60)
    ]
    sorted_rows = agg.select_changed_rows(rows, save_flips_only=False)
    text = agg.format_changed_rows(sorted_rows, limit=10)
    assert "and 50 more" in text


def test_format_changed_rows_limit_zero_means_no_limit():
    rows = [_row_with_delta(f"C{i:02d}", 0.1 + i * 0.001) for i in range(20)]
    sorted_rows = agg.select_changed_rows(rows, save_flips_only=False)
    text = agg.format_changed_rows(sorted_rows, limit=0)
    assert "more" not in text  # no truncation marker
    # All 20 row lines present.
    assert sum(1 for line in text.splitlines() if line.startswith("- ")) == 20


# ---------------------------------------------------------------------------
# run() end-to-end with tempdir fixtures
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> argparse.Namespace:
    base = {
        "paths": [],
        "glob": None,
        "discover": None,
        "changed_only": False,
        "save_flips_only": False,
        "limit": 50,
        "json_out": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_end_to_end_no_json_out(capsys, tmp_path):
    p = tmp_path / "shadow_final_judgments.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(_row(candidate_name=n)) for n in ["A", "B", "C"]
        ) + "\n",
        encoding="utf-8",
    )
    args = _make_args(paths=[str(p)])
    rc = agg.run(args)
    out = capsys.readouterr()
    assert rc == 0
    assert "Shadow Judgment Summary" in out.out
    assert "total_compared" in out.out
    # No JSON file written.
    assert not (tmp_path / "summary.json").exists()


def test_run_writes_json_out_matches_in_memory_summary(capsys, tmp_path):
    p = tmp_path / "shadow_final_judgments.jsonl"
    p.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    out_path = tmp_path / "summary.json"
    args = _make_args(paths=[str(p)], json_out=str(out_path))
    rc = agg.run(args)
    capsys.readouterr()
    assert rc == 0
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))

    # Recompute the in-memory summary using the same input and assert equal.
    expected = agg.aggregate(agg.iter_records([p]))
    assert on_disk == expected


def test_run_changed_only_and_save_flips_only_conflict(capsys, tmp_path):
    p = tmp_path / "shadow_final_judgments.jsonl"
    p.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    args = _make_args(
        paths=[str(p)],
        changed_only=True,
        save_flips_only=True,
    )
    rc = agg.run(args)
    err = capsys.readouterr().err
    assert rc == 1
    assert "mutually exclusive" in err


def test_run_no_inputs_returns_1(capsys):
    args = _make_args(paths=[], glob=None)
    rc = agg.run(args)
    err = capsys.readouterr().err
    assert rc == 1
    assert "no input files resolved" in err


def test_run_changed_only_lists_rows(capsys, tmp_path):
    p = tmp_path / "shadow_final_judgments.jsonl"
    rows = [
        _row_with_delta("Big", -0.40),
        _row_with_delta("Mid", 0.20),
        _row(  # not-computed should be excluded from listing
            external_evidence_status="quota_exhausted",
            enriched=None,
            diff={"computed": False, "reason": "quota_exhausted"},
        ),
    ]
    p.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    args = _make_args(paths=[str(p)], changed_only=True, limit=5)
    rc = agg.run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Materially changed cases" in out
    # Big should appear before Mid (larger abs delta).
    big_idx = out.index("Big")
    mid_idx = out.index("Mid")
    assert big_idx < mid_idx


def test_run_json_out_path_is_directory_returns_1(capsys, tmp_path):
    p = tmp_path / "shadow_final_judgments.jsonl"
    p.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    target_dir = tmp_path / "subdir"
    target_dir.mkdir()
    args = _make_args(paths=[str(p)], json_out=str(target_dir))
    with pytest.raises(SystemExit) as exc_info:
        agg.run(args)
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "directory" in err


# ---------------------------------------------------------------------------
# Slice 6: --discover linkedin and headline formatting
# ---------------------------------------------------------------------------


def _patch_state_root(monkeypatch, root: Path) -> None:
    """Patch the ``source_state_root`` symbol *as imported into the tool*.

    The tool does ``from shared.output_paths import source_state_root`` so we
    must patch ``tools.aggregate_shadow_judgments.source_state_root`` (the
    bound name), not ``shared.output_paths.source_state_root``. Standard
    "patch where it's looked up" pattern. The patched function does NOT
    create the directory -- that's deliberate so we can simulate the
    "missing state root" branch.
    """

    def _fake(source: str, *, output_root=None) -> Path:
        return Path(root) / "state" / source

    monkeypatch.setattr(agg, "source_state_root", _fake)


def _write_shadow_file(brief_dir: Path, rows: list[dict]) -> Path:
    brief_dir.mkdir(parents=True, exist_ok=True)
    target = brief_dir / "shadow_final_judgments.jsonl"
    target.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return target


def test_discover_shadow_files_finds_under_state_root(monkeypatch, tmp_path):
    _patch_state_root(monkeypatch, tmp_path)
    state_root = tmp_path / "state" / "linkedin"
    a = _write_shadow_file(state_root / "brief-alpha", [_row(candidate_name="A")])
    b = _write_shadow_file(state_root / "brief-beta", [_row(candidate_name="B")])

    out = agg.discover_shadow_files("linkedin")
    # Both files discovered.
    assert set(p.resolve() for p in out) == {a.resolve(), b.resolve()}
    # Sorted (stable order).
    assert out == sorted(out, key=lambda p: str(p))
    # No duplicates even on a fresh second call.
    out2 = agg.discover_shadow_files("linkedin")
    assert len(out2) == 2


def test_discover_shadow_files_returns_empty_when_state_root_missing(
    monkeypatch, tmp_path
):
    nowhere = tmp_path / "does_not_exist"
    # Patch to a root that has no state/linkedin subtree -- the function must
    # return [] without raising.
    monkeypatch.setattr(
        agg,
        "source_state_root",
        lambda source, *, output_root=None: nowhere / "state" / source,
    )
    out = agg.discover_shadow_files("linkedin")
    assert out == []


def test_discover_shadow_files_rejects_unknown_source():
    with pytest.raises(ValueError):
        agg.discover_shadow_files("github")


def test_run_with_discover_linkedin_flag_aggregates(
    capsys, monkeypatch, tmp_path
):
    _patch_state_root(monkeypatch, tmp_path)
    state_root = tmp_path / "state" / "linkedin"
    _write_shadow_file(
        state_root / "brief-alpha",
        [_row(candidate_name="Alice"), _row(candidate_name="Bob")],
    )

    args = _make_args(discover="linkedin")
    rc = agg.run(args)
    captured = capsys.readouterr()
    out = captured.out

    assert rc == 0
    assert "Discovered 1 shadow files across 1 briefs" in out
    # Headline block with all eight required fields, in order, before the
    # existing summary.
    headline_idx = out.index("=== Headline ===")
    summary_idx = out.index("=== Shadow Judgment Summary ===")
    assert headline_idx < summary_idx
    headline_fields = (
        "total_compared",
        "reject_to_save",
        "save_to_reject",
        "path_only_changes",
        "rationale_only_changes",
        "unavailable_external_evidence",
        "weak_citations",
        "quota_exhausted",
    )
    headline_block = out[headline_idx:summary_idx]
    last_pos = -1
    for field in headline_fields:
        pos = headline_block.find(field)
        assert pos != -1, f"missing headline field: {field}"
        assert pos > last_pos, f"out-of-order headline field: {field}"
        last_pos = pos
    # Existing summary still rendered.
    assert "Shadow Judgment Summary" in out
    assert "total_compared" in out


def test_run_with_discover_linkedin_no_files_exits_zero(
    capsys, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        agg,
        "source_state_root",
        lambda source, *, output_root=None: tmp_path / "missing" / source,
    )
    args = _make_args(discover="linkedin")
    rc = agg.run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "No shadow files found under output/state/linkedin/" in out


def test_run_with_discover_and_explicit_paths_dedupes(
    capsys, monkeypatch, tmp_path
):
    _patch_state_root(monkeypatch, tmp_path)
    state_root = tmp_path / "state" / "linkedin"
    shared_file = _write_shadow_file(
        state_root / "brief-alpha",
        [_row(candidate_name="Alice")],
    )

    # Pass the same file as both explicit AND have it discovered.
    args = _make_args(
        paths=[str(shared_file)],
        discover="linkedin",
        json_out=str(tmp_path / "summary.json"),
    )
    rc = agg.run(args)
    capsys.readouterr()
    assert rc == 0

    # Read the structured summary written by the run itself; if dedup
    # failed, total_rows_read would be 2 (the file processed twice).
    on_disk = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert on_disk["total_rows_read"] == 1
    assert on_disk["total_compared"] == 1


def test_run_with_no_discover_no_paths_no_glob_exits_one(capsys):
    # Empty invocation: no paths, no glob, no --discover -> exit 1 (existing
    # contract preserved by slice 6).
    args = _make_args(paths=[], glob=None, discover=None)
    rc = agg.run(args)
    err = capsys.readouterr().err
    assert rc == 1
    assert "no input files resolved" in err


def test_run_with_discover_preserves_json_out_shape(
    capsys, monkeypatch, tmp_path
):
    _patch_state_root(monkeypatch, tmp_path)
    state_root = tmp_path / "state" / "linkedin"
    _write_shadow_file(
        state_root / "brief-alpha",
        [_row(candidate_name="Alice")],
    )
    out_path = tmp_path / "summary.json"

    args = _make_args(discover="linkedin", json_out=str(out_path))
    rc = agg.run(args)
    capsys.readouterr()
    assert rc == 0
    assert out_path.exists()

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    # No new top-level keys vs. _empty_summary -- discover-mode does NOT
    # fork the JSON schema. Headline is stdout-only.
    expected_keys = set(agg._empty_summary().keys())
    assert set(on_disk.keys()) == expected_keys


def test_format_headline_lists_eight_fields_in_order():
    summary = agg._empty_summary()
    summary["total_compared"] = 7
    summary["reject_to_save"] = 2
    summary["save_to_reject"] = 1
    summary["path_only_changes"] = 3
    summary["rationale_only_changes"] = 4
    summary["unavailable_external_evidence"] = 5
    summary["weak_citations"] = 6
    summary["quota_exhausted"] = 8

    text = agg.format_headline(summary)
    expected_order = (
        "total_compared",
        "reject_to_save",
        "save_to_reject",
        "path_only_changes",
        "rationale_only_changes",
        "unavailable_external_evidence",
        "weak_citations",
        "quota_exhausted",
    )
    last_pos = -1
    for field in expected_order:
        pos = text.find(field)
        assert pos != -1, f"missing field: {field}"
        assert pos > last_pos, f"out-of-order field: {field}"
        last_pos = pos
    # Each value rendered.
    for field, value in (
        ("total_compared", 7),
        ("reject_to_save", 2),
        ("save_to_reject", 1),
        ("path_only_changes", 3),
        ("rationale_only_changes", 4),
        ("unavailable_external_evidence", 5),
        ("weak_citations", 6),
        ("quota_exhausted", 8),
    ):
        assert f"{field}" in text
        assert f": {value}" in text
    # Newline-separated, one field per line under the header.
    lines = [l for l in text.splitlines() if l.strip()]
    # Header + 8 fields = 9 non-empty lines.
    assert len(lines) == 9
    assert lines[0].startswith("=== Headline ===")
