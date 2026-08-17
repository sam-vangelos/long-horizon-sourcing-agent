"""Tests for the Cloris CLI (Slice 1).

These tests do **not** import :mod:`webview` (pywebview) and do **not** bind
real sockets. The CLI is exercised through ``cloris.cli.main`` directly with
``cloris.app.run_app`` monkeypatched to a recorder.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pytest

from cloris import cli as cloris_cli


def _no_pywebview_imported() -> None:
    assert "webview" not in sys.modules, (
        "pywebview must remain lazy; CLI parsing should not import it"
    )
    assert "pywebview" not in sys.modules


def test_root_help_does_not_import_pywebview() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cloris_cli.main(["--help"])
    assert excinfo.value.code == 0
    _no_pywebview_imported()


def test_start_help_does_not_import_pywebview() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cloris_cli.main(["start", "--help"])
    assert excinfo.value.code == 0
    _no_pywebview_imported()


def test_start_invokes_run_app_with_expected_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_app(app: Any, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    sentinel_app = object()

    def fake_create_app() -> Any:
        return sentinel_app

    from cloris import app as cloris_app

    monkeypatch.setattr(cloris_app, "run_app", fake_run_app)
    monkeypatch.setattr(cloris_app, "create_app", fake_create_app)

    rc = cloris_cli.main(["start", "--host", "127.0.0.1", "--port", "0"])
    assert rc == 0

    assert len(calls) == 1
    call = calls[0]
    assert call["app"] is sentinel_app
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 0


def test_parser_does_not_expose_no_window_flag() -> None:
    parser = cloris_cli.build_parser()

    flags: list[str] = []
    for action in _walk_actions(parser):
        flags.extend(action.option_strings)

    assert "--no-window" not in flags, (
        "Cloris must not expose a --no-window CLI flag; testing seams are "
        "Python-level kwargs on cloris.app.run_app only."
    )


def _walk_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """Yield actions across the root parser and any subparsers."""

    actions: list[argparse.Action] = list(parser._actions)
    for action in list(parser._actions):
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                actions.extend(subparser._actions)
    return actions
