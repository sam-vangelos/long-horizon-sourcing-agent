"""Strict JSON extraction from a model reply.

A model asked for JSON sometimes wraps it: a sentence before a markdown fence,
or a bare object followed by "let me know if you need changes". This module
pulls out the one JSON object or array the reply carries and tolerates that
prose. It never repairs: a truncated reply raises instead of being closed up,
because a cut-off document accepted as complete is worse than a failed call.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


def _closing_index(text: str, start: int) -> int:
    """Index of the bracket that closes ``text[start]``, or -1 if it never closes.

    String-aware, so brackets inside JSON strings do not count.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return index
    return -1


def extract_json(text: str) -> dict | list:
    """Return the JSON object or array in ``text``; raise ``json.JSONDecodeError``.

    Takes the body of the first fenced code block if there is one, tries the
    whole (stripped) text, then scans for the first ``{`` or ``[`` from which a
    complete JSON value decodes, so prose before and after is ignored. A
    bracket that never closes means the reply was cut off, and that raises
    rather than falling through to a complete fragment nested inside it.
    """
    candidate = text.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(value, (dict, list)):
            return value

    position = 0
    while True:
        starts = [i for i in (candidate.find("{", position), candidate.find("[", position)) if i != -1]
        if not starts:
            break
        start = min(starts)
        try:
            value, _end = _DECODER.raw_decode(candidate, start)
        except json.JSONDecodeError as exc:
            close = _closing_index(candidate, start)
            if close == -1:
                raise json.JSONDecodeError(
                    "Truncated JSON: bracket never closes", candidate, start
                ) from exc
            # A closed but invalid span (e.g. "[draft]" in prose): skip it whole,
            # so no fragment nested inside a malformed document is returned.
            position = close + 1
            continue
        return value

    raise json.JSONDecodeError("No JSON object or array found", candidate, 0)
