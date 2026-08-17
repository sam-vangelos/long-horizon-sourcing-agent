"""Best-effort introspection of LinkedIn anti-detection posture at session start.

May trigger cooldown's benign expiry cleanup via get_active_backoff().
"""

from __future__ import annotations

import importlib.metadata


def _probe_ghost_cursor() -> tuple[bool, str]:
    try:
        from python_ghost_cursor.playwright_async import create_cursor  # noqa: F401
        return True, "loaded"
    except Exception as exc:
        return False, str(exc)


def describe_posture(
    *,
    input_mode: str | None = None,
    with_decoy: bool = False,
) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []

    try:
        ghost_active, ghost_detail = _probe_ghost_cursor()
    except Exception as exc:
        ghost_active, ghost_detail = False, f"probe failed: {exc}"
    rows.append(("ghost cursor", ghost_active, ghost_detail))

    rows.append(
        (
            "input backend mode",
            input_mode is not None and input_mode != "",
            input_mode or "unset",
        )
    )

    rows.append(
        (
            "decoy",
            bool(with_decoy),
            "enabled" if with_decoy else "dormant",
        )
    )

    from shared import config

    rows.append(
        (
            "cadence pause",
            True,
            f"interval ~{config.CADENCE_INTERVAL_MINUTES} min",
        )
    )

    from shared.governor import (
        MAX_PROFILE_OPENS_PER_24H,
        MAX_PROFILE_OPENS_PER_SESSION,
    )

    rows.append(
        ("MAX_PROFILE_OPENS_PER_SESSION", True, str(MAX_PROFILE_OPENS_PER_SESSION))
    )
    rows.append(("MAX_PROFILE_OPENS_PER_24H", True, str(MAX_PROFILE_OPENS_PER_24H)))

    try:
        from shared import cooldown

        backoff = cooldown.get_active_backoff()
        if backoff is not None:
            rows.append(("forced backoff", True, f"active until {backoff['until']}"))
        else:
            rows.append(("forced backoff", False, "none"))
    except Exception:
        rows.append(("forced backoff", False, "unknown"))

    try:
        version = importlib.metadata.version("rebrowser-playwright")
        rows.append(("driver package", True, f"rebrowser-playwright {version}"))
    except Exception as exc:
        rows.append(("driver package", False, str(exc)))

    return rows


def format_posture(rows: list[tuple[str, bool, str]]) -> list[str]:
    lines: list[str] = []
    for name, is_active, detail in rows:
        tag = "[ok]" if is_active else "[MISSING]"
        lines.append(f"{tag} {name}: {detail}")
    return lines
