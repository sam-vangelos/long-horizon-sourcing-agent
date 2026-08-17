import shared.governor as gov


def test_start_session_explicit_duration_sets_backstop_and_enforces(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(gov.time, "time", lambda: clock["now"])
    monkeypatch.setattr(gov.cooldown, "get_profile_opens_24h", lambda: 0)

    governor = gov.SessionGovernor()
    governor.start_session(session_duration_seconds=3600.0)

    assert governor.session_duration_limit_seconds == 4200.0

    clock["now"] = 1000.0 + 4199.999
    assert governor.check_limits() is None

    clock["now"] = 1000.0 + 4200.001
    assert governor.check_limits().startswith("session_duration")


def test_start_session_without_duration_draws_fresh_legacy_limit_each_call(monkeypatch):
    draws = iter([12600.0, 16200.0])

    def fake_uniform(lower, upper):
        assert lower == gov.GOVERNOR_LEGACY_SESSION_DURATION_MIN_SECONDS
        assert upper == gov.GOVERNOR_LEGACY_SESSION_DURATION_MAX_SECONDS
        return next(draws)

    monkeypatch.setattr(gov.random, "uniform", fake_uniform)
    monkeypatch.setattr(gov.time, "time", lambda: 1000.0)

    governor = gov.SessionGovernor()
    governor.start_session()
    first_limit = governor.session_duration_limit_seconds
    governor.start_session()
    second_limit = governor.session_duration_limit_seconds

    assert 12600.0 <= first_limit <= 16200.0
    assert 12600.0 <= second_limit <= 16200.0
    assert first_limit != second_limit


def test_status_line_renders_real_session_duration_limit(monkeypatch):
    clock = {"now": 5000.0}
    monkeypatch.setattr(gov.time, "time", lambda: clock["now"])
    monkeypatch.setattr(gov.cooldown, "get_profile_opens_24h", lambda: 17)

    governor = gov.SessionGovernor()
    governor.start_session(session_duration_seconds=15190.0)
    clock["now"] = 5000.0 + 3647.0

    assert "Time: 1:00:47/4:23:10" in governor.status_line()
