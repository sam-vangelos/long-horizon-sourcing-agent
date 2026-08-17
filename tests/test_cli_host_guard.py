from __future__ import annotations

import pytest

from cloris import cli as cloris_cli
from cloris.app import NullWindowLauncher, require_loopback_host, run_app


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_require_loopback_host_accepts_loopback(host: str) -> None:
    require_loopback_host(host)


def test_require_loopback_host_rejects_non_loopback() -> None:
    with pytest.raises(
        ValueError,
        match="Cloris has no network-grade auth; non-loopback binds are refused",
    ):
        require_loopback_host("0.0.0.0")


def test_run_app_rejects_non_loopback_before_socket_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_factory_called = False

    def fail_server_factory(*_args: object) -> None:
        nonlocal server_factory_called
        server_factory_called = True
        pytest.fail("server_factory reached")

    monkeypatch.setattr(
        "cloris.app._resolve_port",
        lambda *_args: pytest.fail("_resolve_port reached"),
    )

    with pytest.raises(
        ValueError,
        match="Cloris has no network-grade auth; non-loopback binds are refused",
    ):
        run_app(
            object(),
            host="0.0.0.0",
            port=0,
            launcher=NullWindowLauncher(),
            server_factory=fail_server_factory,
            ensure_chrome=lambda: pytest.fail("ensure_chrome reached"),
        )

    assert server_factory_called is False


def test_start_rejects_non_loopback_host(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cloris_cli.main(["start", "--host", "0.0.0.0"])

    assert excinfo.value.code == 2
    assert capsys.readouterr().err == (
        "cloris: Cloris has no network-grade auth; "
        "non-loopback binds are refused\n"
    )
