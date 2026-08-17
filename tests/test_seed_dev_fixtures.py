"""Tests for the dev-fixture seeder at ``tools/seed_dev_fixtures``.

The seeder lands synthetic state across runtime_state SQLite +
orchestration SQLite + per-run JSONL logs + Designer market-
intelligence artifacts + intake_sessions so the dev server has data on
a fresh checkout. This commit pins all five fixtures: Briefs 1 + 2 + 3
(published) and Briefs 4 + 5 (in-flight intake drafts).

Pinned for Brief 1 (LinkedIn-only):
- 30 candidates with the prescribed save / no / borderline distribution.
- Two saved candidates carry recruiter judgment_accuracy markers.
- One reflection_sessions row at Gate 2 with three pending hunks.
- run_log.jsonl + cost_rollup.json + run-manifest.json are emitted.

Pinned for Brief 2 (multi-module):
- LinkedIn / Researcher / GitHub state-dirs each carry a completed run
  + per-source candidate counts (20 / 18 / 12).
- The orchestration store carries one ``chief_of_staff_runs`` row with
  ``handoff_payloads_json`` populated for every source (top_saves +
  per_source_signal_summary + confidence + candidate_count + save_count).
- Two ``cross_brief_playbook_observations`` rows tied to the synthetic
  principal.
- Researcher carries one ORCID-clean save and one identity-collision
  borderline; GitHub carries the three maintainership levels.

Plus:
- Every fixture brief.json passes V2 schema validation.
- ``seed_all()`` is idempotent across re-invocation.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from shared.brief_v2_schema import validate_v2_brief
from shared.output_paths import (
    designer_state_key,
    github_state_key,
    derive_brief_id,
    researcher_state_key,
    resolve_intake_db_path,
    resolve_orchestration_db_path,
    source_state_root,
)

from tools.seed_dev_fixtures import (
    BRIEF_1_SLUG,
    BRIEF_2_SLUG,
    BRIEF_3_SLUG,
    REPO_ROOT,
    seed_all,
)


@pytest.fixture(scope="module")
def seeded() -> dict:
    """Run the full seeder once per test module and return its summary."""

    return seed_all()


@pytest.fixture(scope="module")
def seeded_again(seeded: dict) -> dict:
    """Re-invoke the seeder and return the second-pass summary."""

    return seed_all()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [BRIEF_1_SLUG, BRIEF_2_SLUG, BRIEF_3_SLUG],
)
def test_fixture_brief_passes_v2_schema_validation(slug: str) -> None:
    brief_path = REPO_ROOT / "config" / slug / "brief.json"
    validate_v2_brief(json.loads(brief_path.read_text()))


# ---------------------------------------------------------------------------
# Brief 1 — LinkedIn-only
# ---------------------------------------------------------------------------


def test_brief_1_seeds_thirty_candidates_with_save_distribution(
    seeded: dict,
) -> None:
    state_key = derive_brief_id(
        brief_path=REPO_ROOT / "config" / BRIEF_1_SLUG / "brief.json"
    )
    db = source_state_root("linkedin") / state_key / "runtime_state.sqlite3"
    assert db.exists(), f"runtime_state.sqlite3 missing at {db}"

    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM candidates WHERE brief_id = ?",
            (state_key,),
        ).fetchone()["c"]
        saves = conn.execute(
            "SELECT COUNT(*) AS c FROM candidates WHERE brief_id = ? "
            "AND terminal_decision = 'SAVE'",
            (state_key,),
        ).fetchone()["c"]
        borderlines = conn.execute(
            "SELECT COUNT(*) AS c FROM candidates WHERE brief_id = ? "
            "AND terminal_decision = 'FACIAL_BORDERLINE'",
            (state_key,),
        ).fetchone()["c"]
        nos = conn.execute(
            "SELECT COUNT(*) AS c FROM candidates WHERE brief_id = ? "
            "AND terminal_decision = 'FACIAL_NO'",
            (state_key,),
        ).fetchone()["c"]

    assert total == 30
    assert saves == 5
    assert borderlines == 5
    assert nos == 20


def test_brief_1_judgment_accuracy_markers_land_on_two_candidates(
    seeded: dict,
) -> None:
    state_key = derive_brief_id(
        brief_path=REPO_ROOT / "config" / BRIEF_1_SLUG / "brief.json"
    )
    db = source_state_root("linkedin") / state_key / "runtime_state.sqlite3"
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT judgment_accuracy, COUNT(*) FROM candidates "
            "WHERE brief_id = ? AND judgment_accuracy IS NOT NULL "
            "GROUP BY judgment_accuracy",
            (state_key,),
        ).fetchall()
    assert dict(rows) == {"useful": 1, "wrong": 1}


def test_brief_1_reflection_session_at_gate_2(seeded: dict) -> None:
    state_key = derive_brief_id(
        brief_path=REPO_ROOT / "config" / BRIEF_1_SLUG / "brief.json"
    )
    db = source_state_root("linkedin") / state_key / "runtime_state.sqlite3"
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT current_phase, state_json FROM reflection_sessions "
            "WHERE brief_id = ? ORDER BY id DESC LIMIT 1",
            (state_key,),
        ).fetchone()
    assert row is not None
    phase, state_json = row
    assert phase == "awaiting_diff"
    assert len(json.loads(state_json)["proposed_hunks"]) == 3


def test_brief_1_run_artifacts_emitted(seeded: dict) -> None:
    state_key = derive_brief_id(
        brief_path=REPO_ROOT / "config" / BRIEF_1_SLUG / "brief.json"
    )
    state_dir = source_state_root("linkedin") / state_key
    assert (state_dir / "run_log.jsonl").exists()

    runs_root = REPO_ROOT / "output" / "runs" / "linkedin" / state_key
    assert runs_root.exists()
    run_dirs = list(runs_root.iterdir())
    assert run_dirs
    run_dir = run_dirs[0]
    assert (run_dir / "cost_rollup.json").exists()
    assert (run_dir / "run-manifest.json").exists()
    assert json.loads((run_dir / "cost_rollup.json").read_text())["total_usd"] > 0


# ---------------------------------------------------------------------------
# Brief 2 — multi-module + chief-of-staff
# ---------------------------------------------------------------------------


def test_brief_2_seeds_per_source_state_dirs(seeded: dict) -> None:
    """LinkedIn / Researcher / GitHub state-dirs all populated with the
    expected candidate counts: 20 / 18 / 12."""

    brief_path = REPO_ROOT / "config" / BRIEF_2_SLUG / "brief.json"
    expected = [
        ("linkedin", derive_brief_id(brief_path=brief_path), 20),
        ("researcher", researcher_state_key(brief_path=brief_path), 18),
        ("github", github_state_key(brief_path=brief_path), 12),
    ]
    for source, state_key, expected_count in expected:
        db = source_state_root(source) / state_key / "runtime_state.sqlite3"
        assert db.exists(), f"missing {source} state-dir SQLite at {db}"
        with sqlite3.connect(str(db)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE brief_id = ?",
                (state_key,),
            ).fetchone()[0]
        assert count == expected_count, (
            f"{source} expected {expected_count} candidates, got {count}"
        )


def test_brief_2_chief_of_staff_run_has_handoff_payloads_per_source(
    seeded: dict,
) -> None:
    """The orchestration store carries one chief_of_staff_runs row for
    Brief 2 with handoff_payloads populated for all three modules. Each
    payload must carry the substantive fields — empty stubs would
    defeat the whole point of audit Move #1."""

    db = resolve_orchestration_db_path()
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT brief_id, status, handoff_payloads_json, "
            "synthesis_output_json FROM chief_of_staff_runs "
            "WHERE brief_id = ?",
            ("Head of Applied AI",),
        ).fetchall()

    assert len(rows) == 1, f"expected 1 chief_of_staff_runs row, got {len(rows)}"
    _, status, handoff_json, synth_json = rows[0]
    assert status == "completed"

    handoff = json.loads(handoff_json)
    assert set(handoff.keys()) == {"linkedin", "researcher", "github"}
    for source, payload in handoff.items():
        assert payload["top_saves"], f"{source} top_saves empty"
        assert payload["per_source_signal_summary"], (
            f"{source} per_source_signal_summary empty"
        )
        assert payload["candidate_count"] > 0
        assert 0.0 <= payload["confidence"] <= 1.0

    synth = json.loads(synth_json)
    assert synth["paragraph"]
    assert synth["source"] == "deterministic"


def test_brief_2_cross_brief_observations_populated(seeded: dict) -> None:
    db = resolve_orchestration_db_path()
    with sqlite3.connect(str(db)) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM cross_brief_playbook_observations "
            "WHERE principal_id = ?",
            ("fixture-principal-acme",),
        ).fetchone()[0]
    assert count >= 2


def test_brief_2_researcher_carries_orcid_and_identity_collision_cases(
    seeded: dict,
) -> None:
    state_key = researcher_state_key(
        brief_path=REPO_ROOT / "config" / BRIEF_2_SLUG / "brief.json"
    )
    db = source_state_root("researcher") / state_key / "runtime_state.sqlite3"
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE brief_id = ?",
            (state_key,),
        ).fetchall()

    has_orcid = False
    has_identity_collision = False
    for (payload_json,) in rows:
        payload = json.loads(payload_json)
        if payload.get("orcid"):
            has_orcid = True
        if payload.get("needs_identity_confirmation"):
            has_identity_collision = True
    assert has_orcid, "no Researcher candidate carries an ORCID"
    assert has_identity_collision, (
        "no Researcher candidate flagged for identity confirmation"
    )


def test_brief_2_github_maintainership_classification_levels_present(
    seeded: dict,
) -> None:
    state_key = github_state_key(
        brief_path=REPO_ROOT / "config" / BRIEF_2_SLUG / "brief.json"
    )
    db = source_state_root("github") / state_key / "runtime_state.sqlite3"
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT terminal_payload_json FROM candidates "
            "WHERE brief_id = ? AND terminal_decision = 'SAVE'",
            (state_key,),
        ).fetchall()
    levels = set()
    for (payload_json,) in rows:
        payload = json.loads(payload_json)
        cls = payload.get("maintainership_classification") or {}
        if cls.get("level"):
            levels.add(cls["level"])
    assert {"project_lead", "maintainer", "contributor"} <= levels


# ---------------------------------------------------------------------------
# Brief 3 — Designer
# ---------------------------------------------------------------------------


def test_brief_3_designer_visual_judgment_carries_six_principles(
    seeded: dict,
) -> None:
    """Every Designer candidate's terminal_payload carries a
    ``visual_judgment`` block whose ``principles`` list scores all six
    rubric principles."""

    state_key = designer_state_key(
        brief_path=REPO_ROOT / "config" / BRIEF_3_SLUG / "brief.json"
    )
    db = source_state_root("designer") / state_key / "runtime_state.sqlite3"
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE brief_id = ?",
            (state_key,),
        ).fetchall()
    assert len(rows) == 15

    expected_principles = {
        "Visual hierarchy",
        "Typographic refinement",
        "Compositional balance",
        "Color system coherence",
        "Conceptual strength",
        "Craft execution",
    }
    for (payload_json,) in rows:
        payload = json.loads(payload_json)
        vj = payload.get("visual_judgment") or {}
        principle_names = {p["name"] for p in vj.get("principles", [])}
        assert principle_names == expected_principles


def test_brief_3_cross_check_disagreement_present(seeded: dict) -> None:
    """At least one Designer candidate has a cross_check payload whose
    max_disagreement_anchors > 1, which is what surfaces the MODELS
    DISAGREE eyebrow on the workspace card."""

    state_key = designer_state_key(
        brief_path=REPO_ROOT / "config" / BRIEF_3_SLUG / "brief.json"
    )
    db = source_state_root("designer") / state_key / "runtime_state.sqlite3"
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE brief_id = ?",
            (state_key,),
        ).fetchall()
    cross_check_count = 0
    disagreement_count = 0
    for (payload_json,) in rows:
        payload = json.loads(payload_json)
        cc = (payload.get("visual_judgment") or {}).get("cross_check")
        if cc:
            cross_check_count += 1
            if int(cc.get("max_disagreement_anchors", 0)) > 1:
                disagreement_count += 1
    assert cross_check_count >= 3
    assert disagreement_count >= 1


def test_brief_3_design_market_artifact_emitted(seeded: dict) -> None:
    state_key = designer_state_key(
        brief_path=REPO_ROOT / "config" / BRIEF_3_SLUG / "brief.json"
    )
    artifact = (
        REPO_ROOT / "output" / "market_intelligence" / state_key
        / "design_market.md"
    )
    assert artifact.exists()
    body = artifact.read_text()
    assert "rubric" in body.lower()
    assert "disagree" in body.lower()


def test_brief_3_principle_feedback_and_excluded_assets_seeded(
    seeded: dict,
) -> None:
    """The Designer recruiter-annotation stores carry the seeded markers."""

    state_key = designer_state_key(
        brief_path=REPO_ROOT / "config" / BRIEF_3_SLUG / "brief.json"
    )
    annotations_db = (
        source_state_root("designer") / state_key / "annotations.sqlite3"
    )
    assert annotations_db.exists()
    with sqlite3.connect(str(annotations_db)) as conn:
        excluded_count = conn.execute(
            "SELECT COUNT(*) FROM excluded_assets WHERE revoked_at IS NULL"
        ).fetchone()[0]
        markers = conn.execute(
            "SELECT marker, principle_name FROM principle_feedback"
        ).fetchall()
    assert excluded_count >= 1
    marker_set = {(m, p) for m, p in markers}
    assert ("useful_guidance", "Craft execution") in marker_set
    assert ("wrong_shallow", "Visual hierarchy") in marker_set


# ---------------------------------------------------------------------------
# Briefs 4 + 5 — intake drafts
# ---------------------------------------------------------------------------


def test_intake_drafts_seeded_at_expected_chapters(seeded: dict) -> None:
    """Brief 4 + Drafts 5a/b/c land at the prescribed current_step
    values and role_titles (or NULL for Draft 5a)."""

    db = resolve_intake_db_path()
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT current_step, role_title, state_json
            FROM intake_sessions
            WHERE completed_at IS NULL AND archived_at IS NULL
              AND (
                role_title IN (?, ?, ?)
                OR (role_title IS NULL AND state_json LIKE '%fixture-draft-5a%')
              )
            """,
            ("VP of Engineering", "Director of Product", "Senior Designer"),
        ).fetchall()
    by_step = {(r["current_step"], r["role_title"]) for r in rows}
    assert ("depth_distinction", "VP of Engineering") in by_step
    assert ("welcome", None) in by_step
    assert ("lookalikes", "Director of Product") in by_step
    assert ("where_to_look", "Senior Designer") in by_step


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_seed_all_is_idempotent_across_invocations(
    seeded: dict, seeded_again: dict
) -> None:
    """A second invocation of ``seed_all()`` must produce the same
    per-brief candidate counts — no duplicate rows, no leaked state.

    For Brief 2 specifically: the chief_of_staff_runs row count must
    stay at 1 (drop + reinsert, not append).
    """

    assert seeded["brief_1"]["candidates"] == seeded_again["brief_1"]["candidates"]
    assert seeded["brief_2"]["candidates"] == seeded_again["brief_2"]["candidates"]
    assert seeded["brief_3"]["candidates"] == seeded_again["brief_3"]["candidates"]

    db = resolve_orchestration_db_path()
    with sqlite3.connect(str(db)) as conn:
        cos_count = conn.execute(
            "SELECT COUNT(*) FROM chief_of_staff_runs WHERE brief_id = ?",
            ("Head of Applied AI",),
        ).fetchone()[0]
    assert cos_count == 1
