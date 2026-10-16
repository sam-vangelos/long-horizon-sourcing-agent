"""Regression tests for shared.bias_controls save semantics."""

from shared.bias_controls import (
    AlertType,
    BiasMonitor,
    DecisionRecord,
    is_save_decision,
)


def _full_decision(decision: str, string_id: str = "s1", candidate_id: str = "c1") -> DecisionRecord:
    return DecisionRecord(
        candidate_id=candidate_id,
        string_id=string_id,
        stage="full",
        decision=decision,
        confidence=0.8,
        capability_area=None,
    )


def test_signal_save_counts_as_save():
    assert is_save_decision("SIGNAL_SAVE") is True


def test_signal_save_triggers_consecutive_save_alert():
    monitor = BiasMonitor(max_consecutive_saves=2)
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))
    monitor.record_decision(_full_decision("SIGNAL_SAVE", candidate_id="c2"))

    alerts = monitor.check_alerts("s1")

    assert any(alert.alert_type == AlertType.CONSECUTIVE_SAVES for alert in alerts)


def test_signal_save_counts_in_save_rate_and_summary():
    monitor = BiasMonitor(save_rate_spike_window=3, save_rate_spike_threshold=2 / 3)
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))
    monitor.record_decision(_full_decision("SIGNAL_SAVE", candidate_id="c2"))
    monitor.record_decision(_full_decision("REJECT", candidate_id="c3"))

    alerts = monitor.check_alerts("s1")
    summary = monitor.session_summary()

    assert any(alert.alert_type == AlertType.SAVE_RATE_SPIKE for alert in alerts)
    assert summary["saves"] == 2
    assert summary["per_string"]["s1"]["saves"] == 2


# ---------------------------------------------------------------------------
# Telemetry demotion (2026-07-04 SPL run): the count-based checks are
# observations, not brakes. Severity "pause" no longer exists for them, the
# messages promise no action, and fired alerts persist for the run report.
# ---------------------------------------------------------------------------


def _facial_decision(decision: str, string_id: str = "s1", candidate_id: str = "f1") -> DecisionRecord:
    return DecisionRecord(
        candidate_id=candidate_id,
        string_id=string_id,
        stage="facial",
        decision=decision,
        confidence=None,
        capability_area=None,
    )


def test_consecutive_saves_is_flag_severity_with_observation_message():
    monitor = BiasMonitor(max_consecutive_saves=2)
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))
    monitor.record_decision(_full_decision("SIGNAL_SAVE", candidate_id="c2"))

    alerts = monitor.check_alerts("s1")

    alert = next(a for a in alerts if a.alert_type == AlertType.CONSECUTIVE_SAVES)
    assert alert.severity == "flag"
    assert "Pausing" not in alert.message
    assert "pause" not in alert.message.lower()


def test_save_rate_spike_is_flag_severity_with_observation_message():
    monitor = BiasMonitor(save_rate_spike_window=3, save_rate_spike_threshold=2 / 3)
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))
    monitor.record_decision(_full_decision("SIGNAL_SAVE", candidate_id="c2"))
    monitor.record_decision(_full_decision("REJECT", candidate_id="c3"))

    alerts = monitor.check_alerts("s1")

    alert = next(a for a in alerts if a.alert_type == AlertType.SAVE_RATE_SPIKE)
    assert alert.severity == "flag"
    assert "Pausing" not in alert.message
    assert "pause" not in alert.message.lower()


def test_fired_alerts_persist_with_payload_and_round_trip_checkpoint(tmp_path):
    monitor = BiasMonitor(max_consecutive_saves=2)
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))
    monitor.record_decision(_full_decision("SIGNAL_SAVE", candidate_id="c2"))
    monitor.check_alerts("s1")

    fired = monitor.fired_alert_records
    assert len(fired) == 1
    assert fired[0]["alert_type"] == AlertType.CONSECUTIVE_SAVES
    assert fired[0]["string_id"] == "s1"
    assert fired[0]["severity"] == "flag"
    assert fired[0]["message"]

    path = tmp_path / "bias.json"
    monitor.save_checkpoint(str(path))
    restored = BiasMonitor(max_consecutive_saves=2)
    restored.load_checkpoint(str(path))
    assert restored.fired_alert_records == fired
    # Dedup state round-trips too: the same alert must not re-fire.
    assert restored.check_alerts("s1") == []


def test_load_checkpoint_without_fired_alerts_key_is_backward_compatible(tmp_path):
    monitor = BiasMonitor()
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))
    path = tmp_path / "bias.json"
    monitor.save_checkpoint(str(path))

    import json as _json

    data = _json.loads(path.read_text())
    data.pop("fired_alerts", None)
    path.write_text(_json.dumps(data))

    restored = BiasMonitor()
    restored.load_checkpoint(str(path))
    assert restored.fired_alert_records == []


def test_string_context_reports_counts_rates_and_fired_types():
    monitor = BiasMonitor(max_consecutive_saves=2)
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="f1"))
    monitor.record_decision(_facial_decision("FACIAL_NO", candidate_id="f2"))
    monitor.record_decision(_facial_decision("FACIAL_SKIP", candidate_id="f3"))
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))
    monitor.record_decision(_full_decision("SAVE", candidate_id="c2"))
    monitor.check_alerts("s1")

    ctx = monitor.string_context("s1")

    assert ctx == {
        "full_evals": 2,
        "saves": 2,
        "rejects": 0,
        "save_rate": 1.0,
        # Opens-for-full-eval rate (Option B contract); FACIAL_SKIP excluded.
        "opens_for_full_eval_rate": 0.5,
        "facial_n": 2,
        "fired_alert_types": [AlertType.CONSECUTIVE_SAVES],
    }


def test_string_context_returns_none_for_unknown_string():
    monitor = BiasMonitor()
    assert monitor.string_context("nope") is None


def test_string_context_excludes_session_scoped_parse_failure_type():
    monitor = BiasMonitor(parse_failure_alarm_rate=0.01)
    for i in range(20):
        monitor.record_decision(_full_decision("REJECT", candidate_id=f"c{i}"))
    monitor.record_decision(_full_decision("PARSE_FAILURE", candidate_id="c20"))
    monitor.check_alerts("s1")

    ctx = monitor.string_context("s1")
    assert AlertType.PARSE_FAILURE_RATE not in ctx["fired_alert_types"]
