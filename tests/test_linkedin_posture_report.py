from linkedin.posture_report import describe_posture
from shared import contracts


def test_reports_ghost_cursor_unavailable_with_reason(monkeypatch):
    monkeypatch.setattr(
        "linkedin.posture_report._probe_ghost_cursor",
        lambda: (False, "No module named 'playwright'"),
    )
    rows = describe_posture()
    ghost_row = next(row for row in rows if row[0] == "ghost cursor")
    assert ghost_row[1] is False
    assert "playwright" in ghost_row[2]


def test_reports_every_required_control_row():
    rows = describe_posture(input_mode="concurrent")
    names = {row[0] for row in rows}
    required = {
        "ghost cursor",
        "input backend mode",
        "decoy",
        "cadence pause",
        "MAX_PROFILE_OPENS_PER_SESSION",
        "MAX_PROFILE_OPENS_PER_24H",
        "forced backoff",
        "driver package",
    }
    assert required.issubset(names)
    # CLO-153: the daily session-count cap is removed; a posture row for it
    # would report a control that no longer exists.
    assert "MAX_SESSIONS_PER_DAY" not in names


def test_posture_report_event_is_registered_in_contracts():
    assert "posture_report" in contracts.RUN_LOG_EVENTS


def test_describe_posture_is_fail_soft_when_a_probe_raises(monkeypatch):
    def _boom():
        raise RuntimeError("ghost probe exploded")

    monkeypatch.setattr("linkedin.posture_report._probe_ghost_cursor", _boom)
    rows = describe_posture(input_mode="concurrent")
    assert isinstance(rows, list)
    ghost_row = next(row for row in rows if row[0] == "ghost cursor")
    assert ghost_row[1] is False
    assert "probe failed:" in ghost_row[2]
