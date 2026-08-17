"""Tests for :mod:`market_intelligence.github_reflection` (OSS Maintainers Slice 9).

Covers:

- ``maybe_build_and_persist_github_research_packet`` is a no-op for
  non-github batches (LinkedIn flow unaffected).
- For github batches, the packet is persisted to disk and attached
  to ``batch.research_context``, ``batch.research_input_path``,
  and ``batch.context_quality``.
- ``build_github_ecosystem_narrative`` produces the expected per-
  project / per-level / confidence-histogram rollup from a fixture
  set of save records.
- Empty / malformed inputs degrade to a structurally-valid empty
  narrative without raising.
"""

from __future__ import annotations

import json
from pathlib import Path

from market_intelligence.github_reflection import (
    build_github_ecosystem_narrative,
    maybe_build_and_persist_github_research_packet,
)
from market_intelligence.schema import MarketEvidenceBatch


def _save_record(
    *,
    decision: str = "SAVE",
    level: str = "maintainer",
    confidence: float = 0.7,
    evidence_sources: list[str] | None = None,
    signals: dict | None = None,
) -> dict:
    return {
        "decision": decision,
        "maintainership": {
            "level": level,
            "confidence": confidence,
            "evidence_sources": evidence_sources
            or [
                "merge_authority:kubernetes/kubernetes:5PRs",
                "contributors_file:kubernetes/kubernetes",
            ],
            "signals": signals or {"merge_authority": 1.0, "contributors_file": 1.0},
        },
    }


# ---------------------------------------------------------------------------
# build_github_ecosystem_narrative — pure transform
# ---------------------------------------------------------------------------


def test_empty_judgments_produces_zeroed_narrative() -> None:
    narrative = build_github_ecosystem_narrative(final_judgments=[])
    assert narrative["totals"]["saves_total"] == 0
    assert narrative["totals"]["saves_with_classification"] == 0
    assert narrative["top_projects"] == []
    assert narrative["top_signals"] == []


def test_single_save_produces_per_project_count() -> None:
    judgments = [_save_record()]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    assert narrative["totals"]["saves_total"] == 1
    assert narrative["totals"]["saves_with_classification"] == 1
    assert narrative["top_projects"][0]["project"] == "kubernetes/kubernetes"
    assert narrative["top_projects"][0]["save_count"] == 1
    assert narrative["top_projects"][0]["by_level"] == {"maintainer": 1}


def test_per_level_breakdown_aggregates_across_saves() -> None:
    judgments = [
        _save_record(level="contributor", confidence=0.3),
        _save_record(level="maintainer", confidence=0.7),
        _save_record(level="maintainer", confidence=0.8),
        _save_record(level="project_lead", confidence=0.9),
    ]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    by_level = narrative["top_projects"][0]["by_level"]
    assert by_level == {"contributor": 1, "maintainer": 2, "project_lead": 1}


def test_confidence_histogram_buckets() -> None:
    judgments = [
        _save_record(level="contributor", confidence=0.1),
        _save_record(level="contributor", confidence=0.4),
        _save_record(level="maintainer", confidence=0.6),
        _save_record(level="maintainer", confidence=0.85),
        _save_record(level="project_lead", confidence=0.95),
    ]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)

    assert narrative["confidence_histograms"]["contributor"] == {
        "0.0-0.25": 1,
        "0.25-0.5": 1,
        "0.5-0.75": 0,
        "0.75-1.0": 0,
    }
    assert narrative["confidence_histograms"]["maintainer"] == {
        "0.0-0.25": 0,
        "0.25-0.5": 0,
        "0.5-0.75": 1,
        "0.75-1.0": 1,
    }


def test_top_signals_counts_per_signal_contribution() -> None:
    judgments = [
        _save_record(
            evidence_sources=[
                "merge_authority:kubernetes/kubernetes:3PRs",
                "release_authorship:kubernetes/kubernetes:2",
            ],
            signals={
                "merge_authority": 1.0,
                "release_authorship": 1.5,
            },
        ),
        _save_record(
            evidence_sources=[
                "merge_authority:etcd-io/etcd:1PR",
            ],
            signals={"merge_authority": 0.5},
        ),
    ]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    signal_counts = {s["signal"]: s["count"] for s in narrative["top_signals"]}
    # `merge_authority` shows up 4 times: 2 evidence_sources + 2 signals dict entries.
    assert signal_counts.get("merge_authority", 0) >= 3
    assert "release_authorship" in signal_counts


def test_multiple_target_projects_per_save() -> None:
    """A candidate active on 2 named projects contributes to both rollups."""

    judgments = [
        _save_record(
            evidence_sources=[
                "merge_authority:kubernetes/kubernetes:5PRs",
                "merge_authority:etcd-io/etcd:3PRs",
            ]
        )
    ]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    projects = {p["project"]: p for p in narrative["top_projects"]}
    assert "kubernetes/kubernetes" in projects
    assert "etcd-io/etcd" in projects
    # Both projects credit one save each.
    assert projects["kubernetes/kubernetes"]["save_count"] == 1
    assert projects["etcd-io/etcd"]["save_count"] == 1


def test_non_save_decisions_excluded() -> None:
    judgments = [
        _save_record(decision="REJECT"),
        _save_record(decision="SAVE"),
    ]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    assert narrative["totals"]["saves_total"] == 1


def test_inferential_save_classes_included() -> None:
    judgments = [
        _save_record(decision="INFERENTIAL_SAVE"),
        _save_record(decision="SIGNAL_SAVE"),
        _save_record(decision="TRANSFERABLE_SAVE"),
    ]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    assert narrative["totals"]["saves_total"] == 3


def test_malformed_records_skipped_silently() -> None:
    judgments = [
        {"not": "a save record"},
        _save_record(),
        {"decision": "SAVE", "maintainership": "not-a-dict"},
        {"decision": "SAVE", "maintainership": {"level": "core_team"}},  # unknown level
    ]
    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    assert narrative["totals"]["saves_total"] >= 1


# ---------------------------------------------------------------------------
# maybe_build_and_persist_github_research_packet — wiring contract
# ---------------------------------------------------------------------------


def _make_batch(
    *,
    source: str,
    output_dir: Path,
    final_judgments: list[dict] | None = None,
) -> MarketEvidenceBatch:
    return MarketEvidenceBatch(
        run_ref=f"{source}:fixture",
        source=source,
        output_dir=str(output_dir),
        brief_version="2.0",
        generated_at="2026-05-03T00:00:00Z",
        final_judgments=final_judgments or [],
    )


def test_non_github_batch_is_noop(tmp_path: Path) -> None:
    """LinkedIn / non-github batches pass through unchanged."""

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    batch = _make_batch(source="linkedin", output_dir=output_dir)
    result = maybe_build_and_persist_github_research_packet(batch)

    assert result.research_context is None
    assert result.research_input_path == ""
    # No file written.
    assert not (output_dir / "github-research-input.json").exists()


def test_github_batch_persists_packet_and_attaches_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    batch = _make_batch(
        source="github",
        output_dir=output_dir,
        final_judgments=[_save_record() for _ in range(3)],
    )

    result = maybe_build_and_persist_github_research_packet(batch)

    packet_path = output_dir / "github-research-input.json"
    assert packet_path.exists()
    written = json.loads(packet_path.read_text())
    assert written["context_metadata"]["source"] == "github"
    assert written["context_metadata"]["analysis_provenance"]
    assert "ecosystem_momentum" in written

    assert result.research_context is not None
    assert result.research_input_path == str(packet_path)
    assert result.context_quality in {"empty", "thin", "substantive"}


def test_context_quality_label_thin_for_few_saves(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    batch = _make_batch(
        source="github",
        output_dir=output_dir,
        final_judgments=[_save_record() for _ in range(2)],
    )
    result = maybe_build_and_persist_github_research_packet(batch)
    assert result.context_quality == "thin"


def test_context_quality_label_substantive_for_many_saves(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    batch = _make_batch(
        source="github",
        output_dir=output_dir,
        final_judgments=[_save_record() for _ in range(8)],
    )
    result = maybe_build_and_persist_github_research_packet(batch)
    assert result.context_quality == "substantive"


def test_context_quality_label_empty_for_no_classified_saves(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    batch = _make_batch(
        source="github",
        output_dir=output_dir,
        final_judgments=[],
    )
    result = maybe_build_and_persist_github_research_packet(batch)
    assert result.context_quality == "empty"


def test_terminal_payload_path_also_supported(tmp_path: Path) -> None:
    """Records with maintainership nested under terminal_payload also classify."""

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    nested_record = {
        "decision": "SAVE",
        "terminal_payload": {
            "candidate_record": {
                "maintainership": {
                    "level": "maintainer",
                    "confidence": 0.7,
                    "evidence_sources": ["merge_authority:kubernetes/kubernetes:5PRs"],
                    "signals": {},
                }
            }
        },
    }
    batch = _make_batch(
        source="github",
        output_dir=output_dir,
        final_judgments=[nested_record],
    )
    result = maybe_build_and_persist_github_research_packet(batch)
    narrative = result.research_context["ecosystem_momentum"]
    assert narrative["totals"]["saves_with_classification"] == 1
