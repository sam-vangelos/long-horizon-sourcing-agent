"""write_json must never leave a truncated file behind (CLO-437).

write_json persists resume-critical state (preflight_v2_brief.json,
execution_plan.json, the constraint manifest). A kill or a serialization
error mid-write must leave the previous file intact, never half-written JSON
that the next resume cannot parse.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from shared.storage import write_json


class _Unserializable:
    """json.dump raises TypeError on this only after earlier keys are written."""


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


def test_failed_serialization_leaves_original_file_intact(tmp_path):
    target = tmp_path / "execution_plan.json"
    write_json(target, {"strings": [{"id": 1, "boolean": "a OR b"}]})
    original = target.read_bytes()

    # Keys serialize in order, so "padding" is emitted before "bad" raises.
    poison = {"padding": "x" * 200_000, "bad": _Unserializable()}
    with pytest.raises(TypeError):
        write_json(target, poison)

    assert target.read_bytes() == original
    assert json.loads(target.read_bytes()) == {"strings": [{"id": 1, "boolean": "a OR b"}]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["execution_plan.json"]


def test_write_goes_through_fsynced_sibling_temp_then_replace(tmp_path, monkeypatch):
    target = tmp_path / "state" / "preflight_v2_brief.json"
    data = {"role": "SWE", "must_haves": ["python", "unicode café"], "n": 3}
    events: list[tuple] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def recording_fsync(fd):
        events.append(("fsync", fd))
        return real_fsync(fd)

    def recording_replace(src, dst):
        src_text = open(src).read()
        events.append(("replace", os.fspath(src), os.fspath(dst), src_text))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)

    write_json(target, data)
    write_json(target, data)

    kinds = [event[0] for event in events]
    assert kinds == ["fsync", "replace", "fsync", "replace"]
    replaces = [event for event in events if event[0] == "replace"]
    for _, src, dst, src_text in replaces:
        assert os.path.dirname(src) == str(target.parent)
        assert src != str(target)
        assert dst == str(target)
        assert src_text == json.dumps(data, indent=2)
    # Temp names are unique per call, so concurrent writers never share one.
    assert replaces[0][1] != replaces[1][1]

    assert target.read_text() == json.dumps(data, indent=2)
    assert sorted(p.name for p in target.parent.iterdir()) == ["preflight_v2_brief.json"]


def test_failed_replace_removes_temp_and_keeps_original(tmp_path, monkeypatch):
    target = tmp_path / "constraint_manifest.json"
    write_json(target, {"v": 1})
    original = target.read_bytes()

    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        write_json(target, {"v": 2})

    assert target.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["constraint_manifest.json"]


@pytest.mark.parametrize("mask", [0o022, 0o077, 0o002])
def test_written_file_mode_honors_umask(tmp_path, mask):
    previous = os.umask(mask)
    try:
        assert _umask() == mask
        target = tmp_path / "nested" / "out.json"
        write_json(target, {"ok": True})
        mode = stat.S_IMODE(target.stat().st_mode)
    finally:
        os.umask(previous)
    assert mode == 0o666 & ~mask
