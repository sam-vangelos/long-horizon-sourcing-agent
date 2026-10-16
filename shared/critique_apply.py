"""Apply structured critique edits to a V2 brief dict."""

from __future__ import annotations

import copy
import re
from typing import Any

_PATH_CHUNK = re.compile(r"^([^\.\[]+)(?:\[(\d+)\])?")


def split_field_path(path: str) -> list[tuple[str, int | None]]:
    """Split ``role_title`` or ``capability_areas[0].name`` into segments."""

    s = path.strip()
    if not s:
        raise ValueError("empty field path")
    parts: list[tuple[str, int | None]] = []
    i = 0
    n = len(s)
    while i < n:
        m = _PATH_CHUNK.match(s[i:])
        if not m:
            raise ValueError(f"invalid path segment in {path!r} at offset {i}")
        name, ix_s = m.group(1), m.group(2)
        idx = int(ix_s) if ix_s is not None else None
        parts.append((name, idx))
        i += m.end()
        if i < n:
            if s[i] != ".":
                raise ValueError(f"expected '.' in path {path!r} at offset {i}")
            i += 1
    return parts


def get_field_value(root: dict[str, Any], path: str) -> Any:
    """Read a field-path value; returns None when the path is absent."""

    try:
        parts = split_field_path(path)
    except ValueError:
        return None
    cur: Any = root
    for key, idx in parts:
        if not isinstance(cur, dict):
            return None
        val = cur.get(key)
        if idx is None:
            cur = val
            continue
        if not isinstance(val, list) or idx < 0 or idx >= len(val):
            return None
        cur = val[idx]
    return cur


def apply_critique_edits(
    v2_draft: dict[str, Any],
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a deep copy of ``v2_draft`` with ordered set edits applied."""

    out = copy.deepcopy(v2_draft)
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        field = str(edit.get("field") or "").strip()
        if not field:
            continue
        op = str(edit.get("op") or "set").lower()
        if op != "set":
            raise ValueError(f"unsupported critique op {op!r} for field {field!r}")
        _apply_set(out, field, edit.get("value"))
    return out


def _apply_set(root: dict[str, Any], path: str, value: Any) -> None:
    parts = split_field_path(path)
    cur: Any = root
    for i, (key, idx) in enumerate(parts):
        last = i == len(parts) - 1
        if not isinstance(cur, dict):
            raise ValueError(f"{path}: expected object before {key!r}")
        if idx is None:
            if last:
                cur[key] = value
            else:
                nxt = cur.get(key)
                if not isinstance(nxt, dict):
                    raise ValueError(f"{path}: {key!r} is not an object")
                cur = nxt
            continue
        arr = cur.get(key)
        if not isinstance(arr, list):
            if last:
                arr = []
                cur[key] = arr
            else:
                raise ValueError(f"{path}: {key!r} is not a list")
        while len(arr) <= idx:
            arr.append({})
        if last:
            arr[idx] = value
        else:
            nxt = arr[idx]
            if not isinstance(nxt, dict):
                raise ValueError(f"{path}: {key}[{idx}] is not an object")
            cur = nxt
