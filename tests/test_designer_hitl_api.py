"""Integration tests for Designer D6 HITL feedback API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from designer.recruiter_annotations import (
    ExcludedAssetStore,
    PrincipleFeedbackStore,
)
from designer.run_end import annotations_db_path
from shared.runtime_state.store import RuntimeStateStore


@pytest.fixture()
def designer_state(tmp_path: Path) -> tuple[Path, RuntimeStateStore, int, str]:
    """Seed a minimal Designer state_dir with one SAVE candidate."""

    state_dir = tmp_path / "output" / "state" / "designer" / "test-brief"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="designer",
        brief_id="test-brief",
        output_dir=str(state_dir),
        mode="full",
    )
    identity_key = "cse:example.com/portfolio"
    terminal_payload = {
        "surface_type": "hitl_visual_review",
        "full_decision": {
            "decision": "SAVE",
            "rationale": "Strong portfolio.",
            "confidence": 0.88,
        },
        "visual_judgment": {
            "model": "test",
            "principles": [
                {
                    "name": "Visual hierarchy",
                    "score": 3,
                    "anchor": "excellent",
                    "reasoning": "Strong hierarchy.",
                    "image_ids": [0],
                    "anchor_consistency_pass": True,
                }
            ],
            "overall_verdict": "yes",
            "overall_confidence": 0.88,
            "fallback_reason": "",
            "assets": [
                {
                    "id": 0,
                    "url": "https://example.com/thumb.jpg",
                    "source": "google_cse",
                    "project_title": "Project 1",
                }
            ],
        },
    }
    payload_json = json.dumps(terminal_payload, sort_keys=True)
    now = "2026-05-20T12:00:00+00:00"
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO candidates(
                source, brief_id, identity_key, display_name, profile_url,
                current_lifecycle_state, terminal_decision, terminal_payload_json,
                first_seen_at, last_seen_at,
                notes, user_status, judgment_accuracy, judgment_accuracy_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', NULL, NULL, NULL)
            """,
            (
                "designer", "test-brief", identity_key, "Test Designer",
                "https://example.com/portfolio", "full_terminal", "SAVE",
                payload_json, now, now,
            ),
        )
        conn.commit()
    store.finish_run(run_id, "completed")
    return state_dir, store, run_id, identity_key


def test_principle_feedback_useful_guidance(
    designer_state: tuple[Path, RuntimeStateStore, int, str],
) -> None:
    state_dir, store, _, identity_key = designer_state
    annotations_db = annotations_db_path(state_dir)
    fb_store = PrincipleFeedbackStore(annotations_db)

    from designer.recruiter_annotations import record_designer_principle_feedback

    marker = record_designer_principle_feedback(
        runtime_state_store=store,
        principle_feedback_store=fb_store,
        source="designer",
        brief_id="test-brief",
        identity_key=identity_key,
        principle_name="Visual hierarchy",
        marker="useful_guidance",
    )
    assert marker.marker == "useful_guidance"
    assert marker.principle_name == "Visual hierarchy"

    # The canonical runtime_state store should have judgment_accuracy set.
    with store.connect() as conn:
        row = conn.execute(
            "SELECT judgment_accuracy FROM candidates WHERE identity_key = ?",
            (identity_key,),
        ).fetchone()
    assert row["judgment_accuracy"] == "useful"


def test_principle_feedback_invalid_marker_raises(
    designer_state: tuple[Path, RuntimeStateStore, int, str],
) -> None:
    state_dir, store, _, identity_key = designer_state
    annotations_db = annotations_db_path(state_dir)
    fb_store = PrincipleFeedbackStore(annotations_db)

    from designer.recruiter_annotations import record_designer_principle_feedback

    with pytest.raises(ValueError, match="Unknown feedback marker"):
        record_designer_principle_feedback(
            runtime_state_store=store,
            principle_feedback_store=fb_store,
            source="designer",
            brief_id="test-brief",
            identity_key=identity_key,
            principle_name="Visual hierarchy",
            marker="invalid_marker",
        )


def test_exclude_asset_and_revoke(
    designer_state: tuple[Path, RuntimeStateStore, int, str],
) -> None:
    state_dir, _, _, identity_key = designer_state
    annotations_db = annotations_db_path(state_dir)
    exc_store = ExcludedAssetStore(annotations_db)

    asset_url = "https://example.com/thumb.jpg"

    # Exclude.
    exclusion = exc_store.exclude(
        candidate_identity_key=identity_key,
        asset_url=asset_url,
        reason="Misrepresentative",
    )
    assert exclusion.asset_url == asset_url

    active = exc_store.active_exclusions_for_candidate(identity_key)
    assert len(active) == 1
    assert active[0].asset_url == asset_url

    # Revoke.
    revoked = exc_store.revoke(
        candidate_identity_key=identity_key,
        asset_url=asset_url,
    )
    assert revoked is True

    active_after = exc_store.active_exclusions_for_candidate(identity_key)
    assert len(active_after) == 0


def test_feedback_markers_persist_across_reload(
    designer_state: tuple[Path, RuntimeStateStore, int, str],
) -> None:
    state_dir, store, _, identity_key = designer_state
    annotations_db = annotations_db_path(state_dir)

    fb_store = PrincipleFeedbackStore(annotations_db)
    from designer.recruiter_annotations import record_designer_principle_feedback

    record_designer_principle_feedback(
        runtime_state_store=store,
        principle_feedback_store=fb_store,
        source="designer",
        brief_id="test-brief",
        identity_key=identity_key,
        principle_name="Visual hierarchy",
        marker="wrong_shallow",
        note="Score was too generous.",
    )

    # Re-open the store (simulates page reload / fresh aggregation).
    fb_store_2 = PrincipleFeedbackStore(annotations_db)
    markers = fb_store_2.markers_for_candidate(identity_key)
    assert len(markers) == 1
    assert markers[0].marker == "wrong_shallow"
    assert markers[0].note == "Score was too generous."


def test_excluded_assets_persist_across_reload(
    designer_state: tuple[Path, RuntimeStateStore, int, str],
) -> None:
    state_dir, _, _, identity_key = designer_state
    annotations_db = annotations_db_path(state_dir)
    asset_url = "https://example.com/thumb.jpg"

    exc_store = ExcludedAssetStore(annotations_db)
    exc_store.exclude(
        candidate_identity_key=identity_key,
        asset_url=asset_url,
    )

    # Re-open.
    exc_store_2 = ExcludedAssetStore(annotations_db)
    active = exc_store_2.active_exclusions_for_candidate(identity_key)
    assert len(active) == 1
    assert active[0].asset_url == asset_url
