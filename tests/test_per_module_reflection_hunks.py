"""Per-module reflection hunk composer tests — audit Move #9.

Asserts:

- Researcher hunks fire on a fixture run that demonstrates a clearly
  too-strict h_index floor (most candidates rejected at facial) and
  a clearly under-broad conference allowlist (saves publishing
  outside the declared list).
- GitHub hunks fire when saved candidates' evidence cites projects
  outside the declared target_projects, OR when classifier outputs
  cluster below the declared maintainership floor.
- Exec Search hunks fire when the brief carries a small
  company_stage_signals list and saves_total is thin relative to
  surfaced_total.
- All hunks land in the canonical Gate-2 dict shape (hunk_id,
  section, kind, label, before, after, rationale, confidence,
  default_approved, target_field) so reflection.py can project them
  through the existing hunks pipeline without per-module routing.
- All composers return [] cleanly when the run produced zero
  decisions (failure-mode posture for the wider Gate-2 flow).
"""

from __future__ import annotations

from market_intelligence.exec_search_reflection import propose_exec_search_hunks
from market_intelligence.github_reflection import propose_github_hunks
from market_intelligence.researcher_reflection import propose_researcher_hunks


# ---------------------------------------------------------------------------
# Hunk dict shape contract (Gate-2 propose-phase)
# ---------------------------------------------------------------------------


_REQUIRED_HUNK_KEYS = frozenset(
    {
        "hunk_id",
        "section",
        "kind",
        "label",
        "before",
        "after",
        "rationale",
        "confidence",
        "default_approved",
        "target_field",
    }
)


def _assert_hunk_shape(hunk: dict) -> None:
    """Pin the hunk dict against the Gate-2 propose-phase contract."""

    assert isinstance(hunk, dict)
    missing = _REQUIRED_HUNK_KEYS - hunk.keys()
    assert not missing, f"hunk missing required keys: {missing}"
    assert isinstance(hunk["hunk_id"], str) and hunk["hunk_id"]
    assert isinstance(hunk["section"], str) and hunk["section"]
    assert hunk["kind"] in {"add", "modify", "remove", "rubric_refine"}
    assert isinstance(hunk["label"], str) and hunk["label"]
    assert hunk["before"] is None or isinstance(hunk["before"], str)
    assert isinstance(hunk["after"], str) and hunk["after"]
    assert isinstance(hunk["rationale"], str) and hunk["rationale"]
    assert isinstance(hunk["confidence"], (int, float))
    assert 0.0 <= hunk["confidence"] <= 1.0
    assert isinstance(hunk["default_approved"], bool)
    assert isinstance(hunk["target_field"], str) and hunk["target_field"]


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


def _researcher_facial_no(*, h_index: int, name: str = "Cand") -> dict:
    return {
        "decision": "FACIAL_NO",
        "candidate": {
            "name": name,
            "h_index": h_index,
            "papers_in_window": 1,
            "affiliations": ["MIT (US)"],
            "top_venues": [],
        },
    }


def _researcher_save(
    *,
    h_index: int,
    name: str = "Cand",
    venues: list[str] | None = None,
) -> dict:
    return {
        "decision": "SAVE",
        "candidate": {
            "name": name,
            "h_index": h_index,
            "papers_in_window": 6,
            "affiliations": ["MIT (US)"],
            "top_venues": venues or ["NeurIPS"],
        },
    }


def test_researcher_hunks_fire_on_dense_facial_no_at_floor():
    brief = {
        "source_config": {
            "researcher": {
                "h_index_floor": 12,
                "conference_allowlist": ["NeurIPS", "ICML"],
                "research_topics": ["RLHF", "agent infra", "inference systems"],
            }
        }
    }
    judgments = (
        [_researcher_facial_no(h_index=8) for _ in range(15)]
        + [_researcher_save(h_index=14) for _ in range(3)]
    )
    hunks = propose_researcher_hunks(final_judgments=judgments, brief_raw=brief)

    assert len(hunks) >= 1
    for hunk in hunks:
        _assert_hunk_shape(hunk)
    sections = {h["section"] for h in hunks}
    assert "h_index_floor" in sections


def test_researcher_hunks_propose_expanded_allowlist_when_off_list_venues_dominate():
    brief = {
        "source_config": {
            "researcher": {
                "conference_allowlist": ["NeurIPS"],
            }
        }
    }
    # Saved candidates cluster at ICML / ICLR — not in the allowlist.
    judgments = [
        _researcher_save(h_index=14, venues=["ICML", "ICLR"]) for _ in range(8)
    ] + [_researcher_save(h_index=15, venues=["NeurIPS"]) for _ in range(2)]
    hunks = propose_researcher_hunks(final_judgments=judgments, brief_raw=brief)

    sections = {h["section"] for h in hunks}
    assert "conference_allowlist" in sections
    allowlist_hunk = next(h for h in hunks if h["section"] == "conference_allowlist")
    assert "ICML" in allowlist_hunk["after"]
    assert "ICLR" in allowlist_hunk["after"]
    _assert_hunk_shape(allowlist_hunk)


def test_researcher_hunks_empty_on_empty_input():
    assert propose_researcher_hunks(final_judgments=[]) == []
    assert propose_researcher_hunks(final_judgments=None) == []


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def _github_save_with_evidence(
    *,
    project: str,
    level: str,
    confidence: float = 0.8,
    extra_signals: dict | None = None,
) -> dict:
    """A SAVE-class final-judgment row with maintainership classification.

    Mirrors the shape ``github_reflection._iter_save_records`` reads.
    """

    signals = {"commit_count": 50, "merge_authority": 1, **(extra_signals or {})}
    return {
        "decision": "SAVE",
        "maintainership": {
            "level": level,
            "confidence": confidence,
            "signals": signals,
            "evidence_sources": [
                f"commit_count:{project}",
                f"merge_authority:{project}",
            ],
        },
    }


def test_github_hunks_propose_broaden_target_projects_when_saves_cite_off_list_repos():
    brief = {"target_projects": ["kubernetes/kubernetes"]}
    judgments = [
        _github_save_with_evidence(project="rust-lang/rust", level="maintainer")
        for _ in range(6)
    ] + [
        _github_save_with_evidence(
            project="kubernetes/kubernetes", level="maintainer"
        )
        for _ in range(3)
    ]
    hunks = propose_github_hunks(final_judgments=judgments, brief_raw=brief)

    sections = {h["section"] for h in hunks}
    assert "target_projects" in sections
    broaden = next(h for h in hunks if h["section"] == "target_projects")
    assert "rust-lang/rust" in broaden["after"]
    _assert_hunk_shape(broaden)


def test_github_hunks_propose_lower_threshold_when_saves_cluster_below_floor():
    brief = {
        "target_projects": ["kubernetes/kubernetes"],
        "maintainership_level": "project_lead",
    }
    # Saves cluster at "maintainer" and "contributor" — below the
    # declared "project_lead" floor, but the classifier let them
    # through (e.g., generous confidence threshold). The composer
    # should propose lowering the floor by one rung.
    judgments = (
        [
            _github_save_with_evidence(
                project="kubernetes/kubernetes", level="maintainer"
            )
            for _ in range(5)
        ]
        + [
            _github_save_with_evidence(
                project="kubernetes/kubernetes", level="contributor"
            )
            for _ in range(3)
        ]
    )
    hunks = propose_github_hunks(final_judgments=judgments, brief_raw=brief)

    sections = {h["section"] for h in hunks}
    assert "maintainership_level" in sections
    floor_hunk = next(h for h in hunks if h["section"] == "maintainership_level")
    assert floor_hunk["before"] == "project_lead"
    assert floor_hunk["after"] == "maintainer"
    _assert_hunk_shape(floor_hunk)


def test_github_hunks_empty_on_empty_input():
    assert propose_github_hunks(final_judgments=[]) == []
    assert propose_github_hunks(final_judgments=None) == []


def test_github_hunks_empty_when_no_off_list_evidence():
    brief = {"target_projects": ["kubernetes/kubernetes"]}
    # All saves are on-list — no broaden hunk should fire.
    judgments = [
        _github_save_with_evidence(
            project="kubernetes/kubernetes", level="maintainer"
        )
        for _ in range(10)
    ]
    hunks = propose_github_hunks(final_judgments=judgments, brief_raw=brief)
    sections = {h["section"] for h in hunks}
    assert "target_projects" not in sections


# ---------------------------------------------------------------------------
# Exec Search
# ---------------------------------------------------------------------------


def test_exec_search_hunks_widen_stages_on_thin_save_rate():
    brief = {
        "company_stage_signals": ["growth_stage"],
    }
    # 12 surfaced candidates, only 2 saves — thin rate at the
    # declared stage. Composer should propose widening.
    judgments = [{"decision": "FACIAL_NO"} for _ in range(10)] + [
        {"decision": "SAVE"} for _ in range(2)
    ]
    hunks = propose_exec_search_hunks(final_judgments=judgments, brief_raw=brief)

    sections = {h["section"] for h in hunks}
    assert "company_stage_signals" in sections
    stage_hunk = next(h for h in hunks if h["section"] == "company_stage_signals")
    assert "growth_stage" in stage_hunk["before"]
    assert stage_hunk["after"] != stage_hunk["before"]
    _assert_hunk_shape(stage_hunk)


def test_exec_search_hunks_empty_on_empty_input():
    assert propose_exec_search_hunks(final_judgments=[]) == []
    assert propose_exec_search_hunks(final_judgments=None) == []


def test_exec_search_hunks_empty_when_brief_has_no_stage_signals():
    """No stage signals declared ⇒ no widening hunk to propose."""

    judgments = [{"decision": "FACIAL_NO"} for _ in range(10)] + [
        {"decision": "SAVE"} for _ in range(2)
    ]
    hunks = propose_exec_search_hunks(final_judgments=judgments, brief_raw={})
    assert hunks == []


# ---------------------------------------------------------------------------
# Cross-module: shape contract holds for every emitted hunk
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Audit Move #25: _emit_stage discipline
# ---------------------------------------------------------------------------


def test_researcher_composer_emits_stage_lifecycle(capsys):
    """The researcher composer must emit reflection.researcher:start /
    narrative_built / end stage lines via the briefing_polish-style
    `_emit_stage` so cross-module run traces interleave cleanly."""

    brief = {
        "source_config": {
            "researcher": {
                "h_index_floor": 12,
                "conference_allowlist": ["NeurIPS"],
            }
        }
    }
    judgments = [_researcher_facial_no(h_index=8) for _ in range(10)] + [
        _researcher_save(h_index=14, venues=["ICML"]) for _ in range(5)
    ]
    propose_researcher_hunks(final_judgments=judgments, brief_raw=brief)

    err = capsys.readouterr().err
    assert "[market-intel]" in err
    assert "reflection.researcher:start judgments=15" in err
    assert "reflection.researcher:narrative_built" in err
    assert "reflection.researcher:end" in err
    # End stage must report counted hunks + sections (so the trace
    # tells the operator at-a-glance what fired).
    assert "hunks_proposed=" in err
    assert "sections=" in err


def test_researcher_composer_emits_empty_input_stage(capsys):
    propose_researcher_hunks(final_judgments=[])
    err = capsys.readouterr().err
    assert "reflection.researcher:start judgments=0 result=empty_input" in err


def test_github_composer_emits_stage_lifecycle(capsys):
    brief = {"target_projects": ["kubernetes/kubernetes"]}
    judgments = [
        _github_save_with_evidence(project="rust-lang/rust", level="maintainer")
        for _ in range(5)
    ]
    propose_github_hunks(final_judgments=judgments, brief_raw=brief)

    err = capsys.readouterr().err
    assert "reflection.github:start judgments=5" in err
    assert "reflection.github:narrative_built" in err
    assert "reflection.github:end" in err


def test_github_composer_emits_empty_input_stage(capsys):
    propose_github_hunks(final_judgments=[])
    err = capsys.readouterr().err
    assert "reflection.github:start judgments=0 result=empty_input" in err


def test_github_composer_emits_zero_saves_end_stage(capsys):
    """When the run has judgments but no saves, the composer should
    still emit a clean lifecycle (start + end) with a reason field."""

    judgments = [{"decision": "FACIAL_NO"} for _ in range(5)]
    propose_github_hunks(final_judgments=judgments, brief_raw={})
    err = capsys.readouterr().err
    assert "reflection.github:start judgments=5" in err
    assert "reflection.github:end hunks_proposed=0 reason=zero_saves" in err


def test_exec_search_composer_emits_stage_lifecycle(capsys):
    brief = {"company_stage_signals": ["growth_stage"]}
    judgments = [{"decision": "FACIAL_NO"} for _ in range(10)] + [
        {"decision": "SAVE"} for _ in range(2)
    ]
    propose_exec_search_hunks(final_judgments=judgments, brief_raw=brief)

    err = capsys.readouterr().err
    assert "reflection.exec_search:start judgments=12" in err
    assert "reflection.exec_search:end" in err
    assert "surfaced=12" in err
    assert "saves=2" in err


def test_exec_search_composer_emits_empty_input_stage(capsys):
    propose_exec_search_hunks(final_judgments=[])
    err = capsys.readouterr().err
    assert "reflection.exec_search:start judgments=0 result=empty_input" in err


def test_all_per_module_hunks_share_the_gate2_propose_shape():
    """A multi-source fixture run produces hunks across all three
    composers; each one must conform to the Gate-2 propose-phase shape
    so reflection.py can project them through one pipeline."""

    researcher_brief = {
        "source_config": {
            "researcher": {
                "h_index_floor": 12,
                "conference_allowlist": ["NeurIPS"],
            }
        }
    }
    researcher_judgments = [
        _researcher_facial_no(h_index=8) for _ in range(10)
    ] + [_researcher_save(h_index=14, venues=["ICML"]) for _ in range(5)]
    github_brief = {"target_projects": ["kubernetes/kubernetes"]}
    github_judgments = [
        _github_save_with_evidence(project="rust-lang/rust", level="maintainer")
        for _ in range(5)
    ]
    exec_brief = {"company_stage_signals": ["growth_stage"]}
    exec_judgments = [{"decision": "FACIAL_NO"} for _ in range(10)] + [
        {"decision": "SAVE"} for _ in range(2)
    ]

    all_hunks = (
        propose_researcher_hunks(
            final_judgments=researcher_judgments, brief_raw=researcher_brief
        )
        + propose_github_hunks(
            final_judgments=github_judgments, brief_raw=github_brief
        )
        + propose_exec_search_hunks(
            final_judgments=exec_judgments, brief_raw=exec_brief
        )
    )
    assert len(all_hunks) >= 3, f"expected at least one hunk per module; got {all_hunks}"
    for hunk in all_hunks:
        _assert_hunk_shape(hunk)
