"""Researcher Slice 7 — brief polish coverage.

Pins:
- Heuristic seeder promotes `where_to_look.{research_topics,
  conference_allowlist, discipline}` into
  `v2_draft.source_config.researcher`.
- Heuristic does NOT seed explicit floor overrides (those resolve at
  evaluation time per Spec Opinion 7).
- `_research_topics_drift` fires when LLM drops or mutates any
  preserved researcher field.
- `_research_topics_drift` returns None when seed has nothing to
  preserve (classic non-researcher brief).
"""

from __future__ import annotations

from market_intelligence.brief_polish import (
    HeuristicBriefPolishBackend,
    _research_topics_drift,
)


# ---------------------------------------------------------------------------
# Heuristic seeder — researcher source_config promotion
# ---------------------------------------------------------------------------


def test_heuristic_seeds_researcher_source_config_from_chapter_captures() -> None:
    backend = HeuristicBriefPolishBackend()
    result = backend.polish(
        chapter_captures={
            "role": {"title": "Frontier-lab Researcher"},
            "good_looks": {
                "prose": "Original research on RLHF / DPO / SFT at canonical venues.",
            },
            "where_to_look": {
                "target_modules": ["researcher"],
                "research_topics": ["RLHF", "agent infrastructure"],
                "conference_allowlist": ["NeurIPS", "ICML", "ICLR"],
                "discipline": "nlp",
            },
        },
    )

    source_config = result.v2_draft.get("source_config") or {}
    researcher_block = source_config.get("researcher") or {}
    assert researcher_block.get("research_topics") == ["RLHF", "agent infrastructure"]
    assert researcher_block.get("conference_allowlist") == [
        "NeurIPS",
        "ICML",
        "ICLR",
    ]
    assert researcher_block.get("discipline") == "nlp"


def test_heuristic_does_not_seed_explicit_floor_overrides() -> None:
    """Per Spec Opinion 7: floors resolve at evaluation time via
    `researcher.discipline_defaults.resolve_floors`. The wizard does
    not surface raw floor inputs in v1; the heuristic seeder mirrors
    that — even if chapter captures somehow carry floor values, they
    don't get promoted into the v2 draft.
    """

    backend = HeuristicBriefPolishBackend()
    result = backend.polish(
        chapter_captures={
            "role": {"title": "Frontier-lab Researcher"},
            "good_looks": {"prose": "Original research."},
            "where_to_look": {
                "discipline": "nlp",
                "h_index_floor": 12,  # Not surfaced by the wizard, but defensive.
            },
        },
    )
    researcher_block = (
        result.v2_draft.get("source_config", {}).get("researcher") or {}
    )
    assert "h_index_floor" not in researcher_block
    assert "papers_in_window_floor" not in researcher_block
    assert researcher_block.get("discipline") == "nlp"


def test_heuristic_omits_researcher_block_when_no_chapter_captures() -> None:
    """A non-researcher brief (no research_topics, no discipline) should
    NOT have an empty `source_config.researcher` block in the seed.
    """

    backend = HeuristicBriefPolishBackend()
    result = backend.polish(
        chapter_captures={
            "role": {"title": "FDE NYC"},
            "good_looks": {"prose": "Forward-deployed engineering."},
            "where_to_look": {
                "target_modules": ["linkedin"],
                "linkedin_project_id": "12345",
            },
        },
    )
    source_config = result.v2_draft.get("source_config") or {}
    assert "researcher" not in source_config
    # LinkedIn block still surfaces.
    assert source_config.get("linkedin", {}).get("project_id") == "12345"


def test_heuristic_seeds_researcher_alongside_linkedin() -> None:
    """A multi-module brief should produce both linkedin + researcher
    blocks under source_config.
    """

    backend = HeuristicBriefPolishBackend()
    result = backend.polish(
        chapter_captures={
            "role": {"title": "Multi-source Role"},
            "good_looks": {"prose": "Both surfaces matter."},
            "where_to_look": {
                "target_modules": ["linkedin", "researcher"],
                "linkedin_project_id": "12345",
                "discipline": "ml_general",
                "research_topics": ["agent infrastructure"],
            },
        },
    )
    source_config = result.v2_draft.get("source_config") or {}
    assert "linkedin" in source_config
    assert "researcher" in source_config
    assert source_config["researcher"]["discipline"] == "ml_general"


# ---------------------------------------------------------------------------
# _research_topics_drift cascade route
# ---------------------------------------------------------------------------


def test_drift_returns_none_when_seed_has_no_researcher_block() -> None:
    """Classic non-researcher brief — drift detector should be a no-op."""

    seeded = {"source_config": {"linkedin": {"project_id": "12345"}}}
    polished = {"source_config": {"linkedin": {"project_id": "12345"}}}
    assert _research_topics_drift(seeded=seeded, polished=polished) is None


def test_drift_returns_none_when_polished_preserves_all_keys() -> None:
    seeded = {
        "source_config": {
            "researcher": {
                "research_topics": ["RLHF", "agent infrastructure"],
                "conference_allowlist": ["NeurIPS", "ICML"],
                "discipline": "nlp",
                "h_index_floor": 8,
            }
        }
    }
    polished = {
        "source_config": {
            "researcher": {
                "research_topics": ["agent infrastructure", "RLHF"],  # Reorder is OK.
                "conference_allowlist": ["ICML", "NeurIPS"],
                "discipline": "nlp",
                "h_index_floor": 8,
            }
        }
    }
    assert _research_topics_drift(seeded=seeded, polished=polished) is None


def test_drift_fires_when_research_topics_dropped() -> None:
    seeded = {
        "source_config": {
            "researcher": {"research_topics": ["RLHF", "DPO"]}
        }
    }
    polished = {"source_config": {"researcher": {"research_topics": ["RLHF"]}}}
    drift = _research_topics_drift(seeded=seeded, polished=polished)
    assert drift is not None
    assert "research_topics" in drift
    assert "DPO" in drift  # The dropped topic should appear in the diagnostic


def test_drift_fires_when_discipline_changed() -> None:
    seeded = {"source_config": {"researcher": {"discipline": "nlp"}}}
    polished = {"source_config": {"researcher": {"discipline": "ml_general"}}}
    drift = _research_topics_drift(seeded=seeded, polished=polished)
    assert drift is not None
    assert "discipline" in drift
    assert "nlp" in drift
    assert "ml_general" in drift


def test_drift_fires_when_explicit_floor_dropped() -> None:
    seeded = {"source_config": {"researcher": {"h_index_floor": 12}}}
    polished = {"source_config": {"researcher": {"discipline": "nlp"}}}
    drift = _research_topics_drift(seeded=seeded, polished=polished)
    assert drift is not None
    assert "h_index_floor" in drift


def test_drift_fires_when_polished_drops_entire_researcher_block() -> None:
    seeded = {
        "source_config": {
            "researcher": {"research_topics": ["RLHF"], "discipline": "nlp"}
        }
    }
    polished = {"source_config": {"linkedin": {"project_id": "12345"}}}
    drift = _research_topics_drift(seeded=seeded, polished=polished)
    assert drift is not None
    # Both research_topics and discipline should appear in the drift report.
    assert "research_topics" in drift
    assert "discipline" in drift


def test_drift_returns_none_when_polished_adds_unrelated_research_keys() -> None:
    """The drift detector preserves SEED-side keys; it doesn't enforce
    that the polished output is a subset of the seed. New keys added by
    the LLM are fine — only loss/mutation of seeded keys is drift.
    """

    seeded = {"source_config": {"researcher": {"discipline": "nlp"}}}
    polished = {
        "source_config": {
            "researcher": {
                "discipline": "nlp",
                "research_topics": ["RLHF"],  # Extra; not in seed.
            }
        }
    }
    assert _research_topics_drift(seeded=seeded, polished=polished) is None
