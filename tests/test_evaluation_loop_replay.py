"""Recorded observable effects gate the evaluation-loop migration."""
import json

import pytest

from evaluation_loop_replay import FIXTURES, assert_recording, replay


@pytest.mark.parametrize("recording_path", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.stem)
def test_evaluation_loop_recording(recording_path, tmp_path):
    recording = json.loads(recording_path.read_text())
    actual = replay(recording["input"], tmp_path)
    assert_recording(recording["effects"], actual, name=recording["name"])
    escaped = next(item["value"] for item in actual if item["effect"] == "escape")
    if recording["name"] == "transient_pagination":
        assert escaped["type"] == "TransientPaginationError"
    elif recording["name"].startswith("checkpoint_"):
        assert escaped["type"] == "RuntimeError"
        assert "recorded interruption" in escaped["message"]
    else:
        assert escaped is None
    if recording["name"] in {"json_truncation_recovery", "provider_truncation_recovery"}:
        requests = [item for item in actual if item["effect"] == "llm.request"]
        assert len(requests) == 2
        attempts = next(item["value"]["rows"] for item in actual
                        if item["effect"] == "runtime.rows"
                        and item["value"]["table"] == "candidate_attempts")
        assert [row["status"] for row in attempts if row["stage"] == "full"] == ["failed", "succeeded"]
        assert sum(item["effect"] == "llm.facial" for item in actual) == 1
    if recording["name"] == "disconnect_reentry":
        assert sum(item["effect"] == "recovery.restore" for item in actual) == 1
    if recording["name"] == "fresh_run":
        assert sum(item["effect"] == "llm.strategy" for item in actual) == 1
    if recording["name"] == "block_adaptation":
        assert sum(item["effect"] == "llm.adapt" for item in actual) == 1
    transaction_depth = 0
    for item in actual:
        if item["effect"] == "runtime.sql":
            command = item["value"].strip().split()[0]
            if command == "BEGIN":
                transaction_depth += 1
            elif command in {"COMMIT", "ROLLBACK"}:
                transaction_depth -= 1
        if item["effect"] == "projection.publish":
            assert transaction_depth == 0, "projection published before canonical commit"
    assert transaction_depth == 0
    if recording["name"].startswith("checkpoint_"):
        units = next(item["value"]["rows"] for item in actual
                     if item["effect"] == "runtime.rows"
                     and item["value"]["table"] == "work_units")
        current = max(units, key=lambda row: row["run_id"])
        cursor = current["checkpoint_json"]["experiment_state"]["variants"]["root"]["allocator_page_cursor"]
        assert cursor == (1 if recording["name"] == "checkpoint_before_commit" else 2)


def test_effect_comparator_detects_missing_and_reordered_writes(capsys):
    expected = [
        {"effect": "runtime.write", "value": "candidate settled"},
        {"effect": "checkpoint", "value": "cursor N+1"},
    ]
    for wrong in (expected[:1], list(reversed(expected))):
        with pytest.raises(AssertionError, match="effect transcript changed"):
            assert_recording(expected, wrong, name="mutation red proof")
    assert "recorded" in capsys.readouterr().out
