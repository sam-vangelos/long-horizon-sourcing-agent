"""shared.notify — the "tell Sam" path for the death watchdog (CLO-243 slice 3).

Pins the contract every caller relies on:

- ``notify_sam`` never raises, whatever its transports do.
- The same ``dedupe_key`` alerts once; a delivery that reached no transport
  is not recorded, so the next scan retries it.
- The macOS transport passes title/body as osascript argv (no shell, no
  string splicing into AppleScript), so quotes cannot break or inject.
- The Slack transport reads the bot token and Sam's user id from the
  Keychain, opens the DM, posts, and never puts the token in argv or logs.
  Absent Keychain items skip Slack with one log line.

No test here touches the real osascript, the real Keychain, or the network.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
from pathlib import Path

import pytest

from shared.notify import notifier
from shared.notify import notify_sam


SECRET = "xoxb-test-secret-token-do-not-log"


@pytest.fixture
def fake_transports(monkeypatch: pytest.MonkeyPatch):
    sent: list[tuple[str, str, str]] = []

    def _mac(title: str, body: str) -> bool:
        sent.append(("macos", title, body))
        return True

    def _slack(text: str) -> bool:
        sent.append(("slack", text, ""))
        return True

    monkeypatch.setattr(notifier, "_send_macos", _mac)
    monkeypatch.setattr(notifier, "_send_slack", _slack)
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    return sent


# --- notify_sam -------------------------------------------------------------


def test_notify_sends_both_transports_and_returns_true(tmp_path: Path, fake_transports) -> None:
    assert notify_sam("died", "Cloris died", "Run 4 is gone.", "k1", state_dir=tmp_path) is True
    kinds = [entry[0] for entry in fake_transports]
    assert kinds == ["macos", "slack"]
    assert "Cloris died" in fake_transports[1][1]
    assert "Run 4 is gone." in fake_transports[1][1]


def test_notify_dedupes_same_key(tmp_path: Path, fake_transports) -> None:
    assert notify_sam("died", "t", "b", "same-key", state_dir=tmp_path) is True
    assert notify_sam("died", "t", "b", "same-key", state_dir=tmp_path) is False
    assert len(fake_transports) == 2  # one macOS + one Slack, once
    assert notifier.already_sent("same-key", state_dir=tmp_path) is True
    assert notifier.already_sent("other-key", state_dir=tmp_path) is False


def test_notify_dedupe_survives_a_new_process_view(tmp_path: Path, fake_transports) -> None:
    notify_sam("stopped", "t", "b", "persisted", state_dir=tmp_path)
    state = json.loads((tmp_path / notifier.DEDUPE_FILENAME).read_text())
    assert "persisted" in state["sent"]
    assert state["sent"]["persisted"]["kind"] == "stopped"


def test_notify_failed_delivery_is_not_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    monkeypatch.setattr(notifier, "_send_macos", lambda title, body: False)
    monkeypatch.setattr(notifier, "_send_slack", lambda text: False)
    assert notify_sam("died", "t", "b", "retry-me", state_dir=tmp_path) is False
    assert notifier.already_sent("retry-me", state_dir=tmp_path) is False


def test_notify_never_raises_when_transports_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("transport exploded")

    monkeypatch.setattr(notifier, "_send_macos", _boom)
    monkeypatch.setattr(notifier, "_send_slack", _boom)
    assert notify_sam("died", "t", "b", "k", state_dir=tmp_path) is False


def test_notify_one_transport_is_enough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    monkeypatch.setattr(notifier, "_send_macos", lambda title, body: True)

    def _boom(text):
        raise RuntimeError("slack down")

    monkeypatch.setattr(notifier, "_send_slack", _boom)
    assert notify_sam("died", "t", "b", "k", state_dir=tmp_path) is True
    assert notifier.already_sent("k", state_dir=tmp_path) is True


def test_notify_retries_a_failed_configured_slack_without_repeating_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Brief: "by Slack DM when configured". A transient Slack failure must not
    # become a silent loss just because the macOS banner landed.
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    calls: list[str] = []
    slack_ok = {"value": False}

    def _mac(title, body):
        calls.append("macos")
        return True

    def _slack(text):
        calls.append("slack")
        return slack_ok["value"]

    monkeypatch.setattr(notifier, "_send_macos", _mac)
    monkeypatch.setattr(notifier, "_send_slack", _slack)

    assert notify_sam("died", "Cloris died", "Run 4", "k", state_dir=tmp_path) is True
    assert calls == ["macos", "slack"]

    # Next scan: only Slack is retried, through the pending queue.
    slack_ok["value"] = True
    assert notifier.retry_pending(state_dir=tmp_path) == 1
    assert calls == ["macos", "slack", "slack"]

    # Delivered everywhere now: nothing pending, the key stays deduped.
    assert notifier.retry_pending(state_dir=tmp_path) == 0
    assert notify_sam("died", "Cloris died", "Run 4", "k", state_dir=tmp_path) is False
    assert calls == ["macos", "slack", "slack"]


def test_notify_same_key_call_also_retries_pending_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    calls: list[str] = []
    monkeypatch.setattr(notifier, "_send_macos", lambda t, b: calls.append("macos") or True)
    results = iter([False, True])
    monkeypatch.setattr(notifier, "_send_slack", lambda text: calls.append("slack") or next(results))

    assert notify_sam("stuck", "t", "b", "k2", state_dir=tmp_path) is True
    assert notify_sam("stuck", "t", "b", "k2", state_dir=tmp_path) is True
    assert calls == ["macos", "slack", "slack"]


def test_notify_unconfigured_slack_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No Keychain items is a choice, not a failure: _send_slack returns None.
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    calls: list[str] = []
    monkeypatch.setattr(notifier, "_send_macos", lambda t, b: calls.append("macos") or True)
    monkeypatch.setattr(notifier, "_send_slack", lambda text: calls.append("slack") or None)

    assert notify_sam("died", "t", "b", "k3", state_dir=tmp_path) is True
    assert notifier.retry_pending(state_dir=tmp_path) == 0
    assert calls == ["macos", "slack"]


def test_notify_pending_retries_are_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    calls: list[str] = []
    monkeypatch.setattr(notifier, "_send_macos", lambda t, b: True)
    monkeypatch.setattr(notifier, "_send_slack", lambda text: calls.append("slack") or False)

    notify_sam("died", "t", "b", "k4", state_dir=tmp_path)
    for _ in range(notifier.MAX_TRANSPORT_RETRIES + 5):
        notifier.retry_pending(state_dir=tmp_path)
    assert len(calls) == 1 + notifier.MAX_TRANSPORT_RETRIES


def test_notify_never_raises_on_unusable_state_dir(
    tmp_path: Path, fake_transports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(notifier, "_fallback_state_dir", lambda: tmp_path / "fallback")
    blocker = tmp_path / "a-file"
    blocker.write_text("not a dir")
    # The state dir path runs through a regular file: nothing can be created.
    result = notify_sam("died", "t", "b", "k", state_dir=blocker / "nested")
    # The alarm still goes out (fail open on dedupe, never on delivery).
    assert result is True
    assert [entry[0] for entry in fake_transports] == ["macos", "slack"]
    # A 2-minute watchdog must not re-send it every scan: dedupe falls back to
    # a temp-dir state file instead of failing open forever.
    assert notify_sam("died", "t", "b", "k", state_dir=blocker / "nested") is False
    assert [entry[0] for entry in fake_transports] == ["macos", "slack"]


def test_notify_never_raises_on_corrupt_dedupe_file(tmp_path: Path, fake_transports) -> None:
    (tmp_path / notifier.DEDUPE_FILENAME).write_text("{not json")
    assert notify_sam("died", "t", "b", "k", state_dir=tmp_path) is True
    assert notifier.already_sent("k", state_dir=tmp_path) is True


def test_notify_disabled_by_env(tmp_path: Path, fake_transports, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(notifier.DISABLE_ENV, "1")
    assert notify_sam("died", "t", "b", "k", state_dir=tmp_path) is False
    assert fake_transports == []


def test_notify_rejects_garbage_arguments_without_raising(tmp_path: Path, fake_transports) -> None:
    assert notify_sam(None, None, None, None, state_dir=tmp_path) is False  # type: ignore[arg-type]
    assert fake_transports == []


# --- macOS transport --------------------------------------------------------


def test_macos_passes_text_as_argv_not_script(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _runner(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs.get("shell") in (None, False)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(notifier.sys, "platform", "darwin")
    title = 'Cloris "stopped"'
    body = 'it said: end run" & do shell script "rm -rf ~'
    assert notifier._send_macos(title, body, runner=_runner) is True
    argv = calls[0]
    assert argv[0] == notifier.OSASCRIPT
    # Title and body appear only as trailing argv items after "--", verbatim.
    assert argv[-3:] == ["--", title, body]
    script = " ".join(argv[:-3])
    assert "rm -rf" not in script
    assert "stopped" not in script


def test_macos_failure_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notifier.sys, "platform", "darwin")

    def _runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "boom")

    assert notifier._send_macos("t", "b", runner=_runner) is False

    def _raises(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 10)

    assert notifier._send_macos("t", "b", runner=_raises) is False


def test_macos_skipped_off_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notifier.sys, "platform", "linux")

    def _runner(argv, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("osascript must not run off macOS")

    # None: the transport does not exist here, so there is nothing to retry.
    assert notifier._send_macos("t", "b", runner=_runner) is None


def test_clean_text_strips_controls_and_leading_dash() -> None:
    assert notifier._clean_text("a\x00b\nc", 50) == "ab c"
    assert not notifier._clean_text("-rf", 50).startswith("-")
    assert len(notifier._clean_text("x" * 500, 100)) == 100


# --- Slack transport --------------------------------------------------------


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _keychain(values: dict[str, str | None], calls: list[list[str]]):
    def _runner(argv, **kwargs):
        calls.append(list(argv))
        account = argv[argv.index("-a") + 1]
        value = values.get(account)
        if value is None:
            return subprocess.CompletedProcess(argv, 44, "", "not found")
        return subprocess.CompletedProcess(argv, 0, value + "\n", "")

    return _runner


def test_slack_opens_dm_then_posts(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    keychain_calls: list[list[str]] = []
    requests: list[tuple[str, dict, dict]] = []

    def _opener(request, timeout=None, context=None):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, dict(request.header_items()), payload))
        if request.full_url.endswith("/conversations.open"):
            body = {"ok": True, "channel": {"id": "D123"}}
        else:
            body = {"ok": True, "ts": "1.0"}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    caplog.set_level(logging.DEBUG, logger="cloris.notify")
    ok = notifier._send_slack(
        "*Cloris died*\nRun 4 is gone.",
        keychain_runner=_keychain({"bot_token": SECRET, "user_id": "U999"}, keychain_calls),
        opener=_opener,
    )
    assert ok is True
    assert [url.rsplit("/", 1)[-1] for url, _, _ in requests] == [
        "conversations.open",
        "chat.postMessage",
    ]
    assert requests[0][2] == {"users": "U999"}
    assert requests[1][2]["channel"] == "D123"
    assert "Run 4 is gone." in requests[1][2]["text"]
    headers = {k.lower(): v for k, v in requests[0][1].items()}
    assert headers["authorization"] == f"Bearer {SECRET}"
    # The token never reaches argv or the log.
    for argv in keychain_calls:
        assert SECRET not in " ".join(argv)
        assert argv[:2] == [notifier.SECURITY, "find-generic-password"]
        assert notifier.KEYCHAIN_SERVICE in argv
    assert SECRET not in caplog.text


def test_slack_skipped_when_keychain_empty(caplog) -> None:
    calls: list[list[str]] = []

    def _opener(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("no token, no network")

    caplog.set_level(logging.INFO, logger="cloris.notify")
    # None, not False: "not configured" is not a failure and is never retried.
    assert notifier._send_slack("hi", keychain_runner=_keychain({}, calls), opener=_opener) is None
    skip_lines = [r for r in caplog.records if "skipped" in r.getMessage()]
    assert len(skip_lines) == 1


def test_slack_api_error_returns_false_without_leaking(caplog) -> None:
    calls: list[list[str]] = []

    def _opener(request, timeout=None, context=None):
        return _FakeResponse(json.dumps({"ok": False, "error": "invalid_auth"}).encode())

    caplog.set_level(logging.DEBUG, logger="cloris.notify")
    ok = notifier._send_slack(
        "hi",
        keychain_runner=_keychain({"bot_token": SECRET, "user_id": "U1"}, calls),
        opener=_opener,
    )
    assert ok is False
    assert "invalid_auth" in caplog.text
    assert SECRET not in caplog.text


def test_slack_network_error_returns_false(caplog) -> None:
    calls: list[list[str]] = []

    def _opener(request, timeout=None, context=None):
        raise OSError(f"connection refused while sending {request.full_url}")

    assert (
        notifier._send_slack(
            "hi",
            keychain_runner=_keychain({"bot_token": SECRET, "user_id": "U1"}, calls),
            opener=_opener,
        )
        is False
    )
    assert SECRET not in caplog.text
