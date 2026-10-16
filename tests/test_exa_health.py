"""Offline tests for the Exa readiness probe (S1).

The async validator is stubbed in every test — this band never touches
the network. Blocker kinds must stay inside ``shared.health_types``'s
three-value Literal.
"""

import asyncio

import pytest

from exa import health as exa_health
from shared.health_types import ReadinessReport


_VALID_KINDS = {"auth", "config", "net"}


def _stub_validate(result):
    async def _fake(api_key):
        return result

    return _fake


def _forbid_validate(monkeypatch):
    async def _fake(api_key):  # pragma: no cover - must never run
        raise AssertionError("readiness probe hit the network")

    monkeypatch.setattr(exa_health, "_async_validate", _fake)


def test_probe_missing_key_is_config_blocker(monkeypatch):
    monkeypatch.setattr(exa_health.exa_config, "EXA_API_KEY", "")
    _forbid_validate(monkeypatch)

    report = exa_health.probe_exa_readiness()

    assert isinstance(report, ReadinessReport)
    assert report.ready is False
    assert len(report.blockers) == 1
    blocker = report.blockers[0]
    assert blocker.kind == "config"
    assert blocker.message == "No Exa API key configured."
    assert "EXA_API_KEY" in blocker.remediation
    assert "dashboard.exa.ai" in blocker.remediation


def test_probe_401_is_auth_blocker(monkeypatch):
    monkeypatch.setattr(
        exa_health,
        "_async_validate",
        _stub_validate(
            (False, "Exa API authentication failed: 401 INVALID_API_KEY", "auth")
        ),
    )

    report = exa_health.probe_exa_readiness(api_key="dummy")

    assert report.ready is False
    blocker = report.blockers[0]
    assert blocker.kind == "auth"
    assert blocker.message == "Exa rejected your API key."
    assert "dashboard.exa.ai" in blocker.remediation


def test_probe_402_is_credits_config_blocker(monkeypatch):
    monkeypatch.setattr(
        exa_health,
        "_async_validate",
        _stub_validate((False, "Exa API refused the call: 402", "credits")),
    )

    report = exa_health.probe_exa_readiness(api_key="dummy")

    assert report.ready is False
    blocker = report.blockers[0]
    # "Out of credits" is a config problem, not an auth one — the key is
    # fine, the account is empty. There is no fourth BlockerKind.
    assert blocker.kind == "config"
    assert blocker.kind in _VALID_KINDS
    assert blocker.message == "Exa account is out of credits."
    assert "Top up" in blocker.remediation


def test_probe_network_failure_is_net_blocker(monkeypatch):
    monkeypatch.setattr(
        exa_health,
        "_async_validate",
        _stub_validate((False, "Cannot connect to host api.exa.ai", "net")),
    )

    report = exa_health.probe_exa_readiness(api_key="dummy")

    assert report.ready is False
    blocker = report.blockers[0]
    assert blocker.kind == "net"
    assert blocker.message == "Cloris couldn't reach the Exa API."
    assert "api.exa.ai" in blocker.remediation


def test_probe_ok_is_ready(monkeypatch):
    monkeypatch.setattr(
        exa_health, "_async_validate", _stub_validate((True, None, None))
    )

    report = exa_health.probe_exa_readiness(api_key="dummy")

    assert report.ready is True
    assert report.blockers == ()


# ---------------------------------------------------------------------------
# The validator itself — every test above stubs it out, so without these
# a validator that always returned ok would keep the whole band green.
# ---------------------------------------------------------------------------

def _install_stub_client(monkeypatch, error):
    """Swap ExaClient for a stub whose validate_credentials raises `error`."""

    import exa.client as exa_client

    seen = {}

    class _StubClient:
        def __init__(self, api_key=None, **kwargs):
            seen["api_key"] = api_key

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def validate_credentials(self):
            seen["validated"] = True
            if error is not None:
                raise error

    # _async_validate imports ExaClient from exa.client at call time.
    monkeypatch.setattr(exa_client, "ExaClient", _StubClient)
    return seen


def test_async_validate_maps_auth_error(monkeypatch):
    from exa.client import ExaAuthError

    seen = _install_stub_client(monkeypatch, ExaAuthError("401 INVALID_API_KEY"))

    ok, err, kind = asyncio.run(exa_health._async_validate("dummy-key"))

    assert ok is False
    assert kind == "auth"
    assert "INVALID_API_KEY" in err
    assert seen["api_key"] == "dummy-key"
    assert seen["validated"] is True


def test_async_validate_maps_credits_error(monkeypatch):
    from exa.client import ExaCreditsError

    _install_stub_client(monkeypatch, ExaCreditsError("402 out of credits"))

    ok, err, kind = asyncio.run(exa_health._async_validate("dummy-key"))

    assert ok is False
    assert kind == "credits"
    assert "credits" in err


def test_async_validate_maps_runtime_error_to_net(monkeypatch):
    _install_stub_client(monkeypatch, RuntimeError("Cannot connect to host api.exa.ai"))

    ok, err, kind = asyncio.run(exa_health._async_validate("dummy-key"))

    assert ok is False
    assert kind == "net"
    assert "api.exa.ai" in err


def test_async_validate_ok(monkeypatch):
    seen = _install_stub_client(monkeypatch, None)

    assert asyncio.run(exa_health._async_validate("dummy-key")) == (True, None, None)
    assert seen["validated"] is True


@pytest.mark.parametrize(
    "failure",
    [
        (False, "boom", "auth"),
        (False, "boom", "credits"),
        (False, "boom", "net"),
        (False, "boom", None),
    ],
)
def test_every_blocker_kind_stays_in_the_literal(monkeypatch, failure):
    monkeypatch.setattr(exa_health, "_async_validate", _stub_validate(failure))

    report = exa_health.probe_exa_readiness(api_key="dummy")

    assert report.ready is False
    assert all(blocker.kind in _VALID_KINDS for blocker in report.blockers)
