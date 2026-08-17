"""Tests for sync per-hub registry probes in :mod:`github.health`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from github.health import probe_crates_registry, probe_npm_registry


def _mock_urlopen(*, status: int = 200, exc: Exception | None = None):
    if exc is not None:
        raise exc

    response = MagicMock()
    response.status = status
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


def test_probe_npm_registry_true_on_200() -> None:
    with patch(
        "github.health.urllib.request.urlopen",
        return_value=_mock_urlopen(status=200),
    ) as urlopen:
        assert probe_npm_registry() is True

    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/-/ping")
    assert "npm hub client" in request.headers["User-agent"]


def test_probe_crates_registry_true_on_200() -> None:
    with patch(
        "github.health.urllib.request.urlopen",
        return_value=_mock_urlopen(status=200),
    ) as urlopen:
        assert probe_crates_registry() is True

    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/api/v1/summary")
    assert "crates hub client" in request.headers["User-agent"]


@pytest.mark.parametrize(
    ("probe_fn", "status"),
    [
        (probe_npm_registry, 503),
        (probe_crates_registry, 404),
    ],
)
def test_probe_returns_false_on_non_200(probe_fn, status: int) -> None:
    with patch(
        "github.health.urllib.request.urlopen",
        return_value=_mock_urlopen(status=status),
    ):
        assert probe_fn() is False


@pytest.mark.parametrize(
    "probe_fn",
    [probe_npm_registry, probe_crates_registry],
)
def test_probe_returns_false_on_network_error(probe_fn) -> None:
    with patch(
        "github.health.urllib.request.urlopen",
        side_effect=OSError("network down"),
    ):
        assert probe_fn() is False
