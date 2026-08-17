from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from shared import console_tee


@pytest.fixture(autouse=True)
def _reset_console_tee(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(console_tee, "_installed", False)
    monkeypatch.setattr(console_tee, "_cleanup", None)
    yield
    if console_tee._cleanup is not None:
        console_tee.atexit.unregister(console_tee._cleanup)
        console_tee._cleanup()


def test_enable_console_tee_rotates_oversized_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "live-console.log"
    rotated_path = tmp_path / "live-console.log.1"
    old_content = "oversized log"
    output_path.write_text(old_content)
    rotated_path.write_text("older generation")
    monkeypatch.setattr(console_tee, "MAX_CONSOLE_LOG_BYTES", 4)

    console_tee.enable_console_tee(tmp_path)

    assert rotated_path.read_text() == old_content
    assert output_path.read_text() == (
        f"[console] Mirroring stdout/stderr to {output_path}\n"
    )


def test_enable_console_tee_does_not_rotate_small_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "live-console.log"
    old_content = "small"
    output_path.write_text(old_content)
    monkeypatch.setattr(console_tee, "MAX_CONSOLE_LOG_BYTES", len(old_content) + 1)

    console_tee.enable_console_tee(tmp_path)

    assert not (tmp_path / "live-console.log.1").exists()
    assert output_path.read_text() == (
        f"[console] Mirroring stdout/stderr to {output_path}\n"
    )
