"""Filter structured edits against recruiter-affirmed field paths."""

from __future__ import annotations


def filter_locked_edits(
    edits: list[dict],
    affirmed_fields: list[str],
) -> tuple[list[dict], list[dict]]:
    """Split edits into allowed vs blocked by affirmed prefix match."""

    locks = [str(field) for field in affirmed_fields if field]
    allowed: list[dict] = []
    blocked: list[dict] = []
    for edit in edits:
        field = str(edit.get("field") or "")
        is_locked = any(
            field == lock or field.startswith(lock + ".") or field.startswith(lock + "[")
            for lock in locks
        )
        (blocked if is_locked else allowed).append(edit)
    return allowed, blocked
